#!/usr/bin/env python3
"""Train the one-sequence scorer, taught by the pair scorer we already have.

The pair model reads the state once per option, which is correct and slow.
This trains the same backbone to give the same answers from one reading, so a
five-option decision costs one pass instead of five.

Two signals, mixed:

* **Gold labels** from the training mix, with cross entropy and a Brier term,
  exactly as the pair model was trained.
* **The pair model's own distribution** (`--teacher`), as a KL term. This is
  the part that matters. The first attempt at a one-sequence head never
  learned from hard labels alone — cross entropy sat at ln(number of options)
  — because nothing told the head how the question and an option relate. A
  teacher that is already right hands it that relation on every row, and a
  correct model distilled into a cheaper shape of itself is a far easier
  problem than learning the task again.

Option order is shuffled on every row. The pair scorer could not be
order-biased by construction, this one can, so the training has to remove the
bias and `--flip-check` measures what is left.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel import teacher as teaching
from gavel.schema import read_jsonl
from gavel.span import SpanScorer, build_tokenizer, encode


def batches(rows, size, rng, shuffle_options=True, pack=1, by_state=None, pack_share=0.0):
    """Batches of sequences; each sequence holds one or more questions.

    Yields `(prepared, targets, chunk, permutations)`. Every element of the
    batch is a list of rows that share a state: one row normally, up to
    `pack` rows when packing. `targets[i][j]` is the gold slot of question
    `j` inside its own options, `permutations[i][j][slot]` the original
    option index that landed in that slot.

    Packing is why this exists: a head that only ever saw one `[Q]` group
    answers three bundled questions at 0.065 accuracy (2026-09-21). The
    training sequences have to look like the serving sequences.
    """
    order = list(range(len(rows)))
    rng.shuffle(order)
    packable = [group for group in (by_state or {}).values() if len(group) >= 2]
    for start in range(0, len(order) - size + 1, size):
        prepared, targets, chunk, permutations = [], [], [], []
        for i in order[start : start + size]:
            if pack > 1 and packable and rng.random() < pack_share:
                group = rng.choice(packable)
                members = [rows[j] for j in rng.sample(group, min(pack, len(group)))]
            else:
                members = [rows[i]]
            questions, labels, perms = [], [], []
            for row in members:
                index = list(range(len(row.options)))
                if shuffle_options:
                    rng.shuffle(index)
                questions.append((row.question, [row.options[k].description for k in index]))
                labels.append(index.index(row.label))
                perms.append(index)
            prepared.append((members[0].state, questions))
            targets.append(labels)
            chunk.append(members)
            permutations.append(perms)
        yield prepared, targets, chunk, permutations


def gold_slots(batch, targets, device):
    """(batch, width) one-hot over option slots, and questions per row."""
    target = torch.zeros(batch.group.shape, device=device)
    counts = torch.zeros(len(targets), device=device)
    for i, (labels, widths) in enumerate(zip(targets, batch.counts)):
        offset = 0
        for label, width in zip(labels, widths):
            target[i, offset + label] = 1.0
            offset += width
        counts[i] = len(labels)
    return target, counts.clamp(min=1.0)


@torch.no_grad()
def teacher_distribution(teacher, rows, device, max_length, width):
    """The pair model's probabilities for every question in every row.

    One tokenizer call and one forward for the whole batch. The first version
    called `teacher.decide()` once per row — eight tokenizations, eight
    forwards and eight device syncs per step, which was 40-50 percent of the
    step — and looked the result up by option text, so two options with the
    same text collapsed into one entry. Scores are indexed by position now,
    one softmax per question, laid out in the slot order `encode` uses.
    """
    premises, hypotheses, segments = [], [], []
    for row, (state, questions) in enumerate(rows):
        offset = 0
        for question, options in questions:
            premise = (state or question).strip()
            for option in options:
                premises.append(premise)
                hypotheses.append(teacher._hypothesis(question, option))
            segments.append((row, offset, len(options)))
            offset += len(options)
    # In blocks: a packed batch of 16 sequences with four questions of five
    # options each is 320 pairs, and one forward over all of them took a
    # 24 GB card down at step 1400 on 2026-09-22.
    logits = []
    for start in range(0, len(premises), 64):
        encoding = teacher.tokenizer(premises[start : start + 64], hypotheses[start : start + 64],
                                     padding=True, truncation=True, max_length=max_length,
                                     return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            logits.append(teacher.model(**encoding).logits[:, 0].float() / teacher.temperature)
    logits = torch.cat(logits)
    padded = torch.zeros(len(rows), width, device=device)
    start = 0
    for row, offset, count in segments:
        padded[row, offset : offset + count] = F.softmax(logits[start : start + count], dim=-1)
        start += count
    return padded


def file_distribution(chunk, permutations, width, device):
    """Jev's probabilities where a row has them, laid out like the slots."""
    soft = torch.zeros(len(chunk), width, device=device)
    has = torch.zeros(len(chunk), width, dtype=torch.bool, device=device)
    for i, (members, perms) in enumerate(zip(chunk, permutations)):
        offset = 0
        for row, index in zip(members, perms):
            teacher = row.meta.get("teacher")
            if teacher and len(teacher) == len(row.options):
                values = torch.tensor([teacher[k] for k in index], device=device)
                soft[i, offset : offset + len(index)] = values
                has[i, offset : offset + len(index)] = True
            offset += len(index)
    return soft, has


def packed_loss(log_probabilities, target, questions, valid, soft, brier_weight, teacher_weight):
    """Cross entropy, Brier and the teacher's KL per question, averaged.

    `target` is one-hot per question over the option slots, `questions` the
    number of questions per row, `soft` a per-question distribution or zero
    where no teacher spoke. Padded slots are zeroed before any product, the
    lesson of `distillation_loss`.
    """
    valid = valid.float()
    safe_log = log_probabilities.masked_fill(valid == 0, 0.0)
    probabilities = safe_log.exp() * valid
    nll = -(target * safe_log).sum(-1) / questions
    brier = (((probabilities - target) ** 2) * valid).sum(-1) / questions
    loss = nll.mean() + brier_weight * brier.mean()
    if soft is not None:
        soft = soft * valid
        kl = (soft * (soft.clamp_min(1e-8).log() - safe_log) * valid).sum(-1) / questions
        loss = loss + teacher_weight * kl.mean()
    return loss


@torch.no_grad()
def evaluate(model, tokenizer, rows, device, max_length, batch_size, flip_check):
    model.eval()
    rng = random.Random(7)
    per_source: dict[str, list[int]] = defaultdict(list)
    flips = total = 0
    for prepared, targets, nested, _ in batches(rows, batch_size, rng, shuffle_options=False):
        chunk = [members[0] for members in nested]
        labels = torch.tensor([t[0] for t in targets])
        batch = encode(tokenizer, prepared, max_length, device)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                  enabled=str(device).startswith("cuda"))
        with autocast:
            choice = model(batch).argmax(dim=-1).cpu()
        for row, predicted, gold in zip(chunk, choice.tolist(), labels.tolist()):
            per_source[row.source].append(int(predicted == gold))
        if flip_check:
            reversed_rows = [(state, [(q, list(reversed(o)))])
                             for state, [(q, o)] in prepared]
            back = encode(tokenizer, reversed_rows, max_length, device)
            with autocast:
                other = model(back).argmax(dim=-1).cpu()
        if flip_check:
            for row, first, second in zip(chunk, choice.tolist(), other.tolist()):
                width = len(row.options)
                flips += int(first != width - 1 - second)
                total += 1
    model.train()
    means = {s: sum(v) / len(v) for s, v in sorted(per_source.items())}
    overall = sum(means.values()) / max(1, len(means))
    return overall, means, (flips / total if total else 0.0)


def reuse_encoder(encoder, checkpoint: str) -> tuple[int, int]:
    """Copy the trained encoder out of a pair checkpoint into the span model.

    The pair scorer wraps `AutoModelForSequenceClassification`, whose keys
    are `model.<base>.<layer>` with `<base>` being `model` for ModernBERT and
    `bert` for BERT. The span scorer wraps the bare `AutoModel`, whose keys
    start at `<layer>`. The first version stripped one prefix and matched
    nothing at all — zero of thirty-nine tensors — and would have distilled
    into an untrained backbone without saying so.

    Two things have to hold: strip as many leading components as it takes
    for a key to exist in the target, and merge the token embedding row by
    row, because the span tokenizer added three marker tokens and the
    matrices differ by three rows.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    target = encoder.state_dict()
    loaded, total = {}, 0
    for key, value in state.items():
        if "classifier" in key or key == "log_temperature":
            continue
        total += 1
        parts = key.split(".")
        for depth in range(1, min(4, len(parts))):
            candidate = ".".join(parts[depth:])
            if candidate not in target:
                continue
            have = target[candidate]
            if have.shape == value.shape:
                loaded[candidate] = value
            elif (have.dim() == value.dim() == 2 and have.shape[1] == value.shape[1]):
                merged = have.clone()
                rows = min(have.shape[0], value.shape[0])
                merged[:rows] = value[:rows]
                loaded[candidate] = merged
            break
    encoder.load_state_dict(loaded, strict=False)
    return len(loaded), total


def distillation_loss(log_probabilities, labels, valid, soft, brier_weight,
                      teacher_weight):
    """Cross entropy, Brier and the teacher's KL, over real option slots only.

    A row with fewer options than the widest in its batch has padded slots
    at minus infinity, and the teacher has zero there. Feeding that to
    `F.kl_div` gives 0 * (-inf) = NaN, and the baseline's distillation ran
    to `loss nan` by step 6300 exactly this way. Padded slots are zeroed
    before any product is formed.
    """
    valid = valid.float()
    loss = F.nll_loss(log_probabilities, labels)
    safe_log = log_probabilities.masked_fill(valid == 0, 0.0)
    probabilities = safe_log.exp() * valid
    target = F.one_hot(labels, num_classes=probabilities.size(-1)).float()
    loss = loss + brier_weight * (((probabilities - target) ** 2) * valid).sum(-1).mean()
    if soft is not None and soft.size(-1) == probabilities.size(-1):
        soft = soft * valid
        kl = soft * (soft.clamp_min(1e-8).log() - safe_log) * valid
        loss = loss + teacher_weight * kl.sum(-1).mean()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="llm-semantic-router/Vela-1.0-Encoder-307M")
    parser.add_argument("--init-backbone-from", default="",
                        help="a trained pair checkpoint whose encoder weights are reused")
    parser.add_argument("--teacher", default="",
                        help="a pair checkpoint that supplies soft targets")
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-file", default="",
                        help="soft labels per row id from scripts/label_with_jev.py; "
                             "rows it covers use these instead of the pair model")
    parser.add_argument("--train", default="data/train_v6.jsonl")
    parser.add_argument("--dev", default="data/dev_strat_v2.jsonl")
    parser.add_argument("--out", default="runs/span")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--warmup", type=float, default=0.06)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--eval-rows", type=int, default=2000)
    parser.add_argument("--max-options", type=int, default=12)
    parser.add_argument("--pack", type=int, default=4,
                        help="up to this many questions that share a state in one sequence")
    parser.add_argument("--pack-share", type=float, default=0.5,
                        help="share of sequences that are packed when a state has several rows")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--adam8bit", action="store_true")
    parser.add_argument("--flip-check", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = [r for r in read_jsonl(args.train)
            if r.label is not None and 2 <= len(r.options) <= args.max_options]
    dev = [r for r in read_jsonl(args.dev)
           if r.label is not None and 2 <= len(r.options) <= args.max_options]
    rng.shuffle(dev)
    dev = dev[: args.eval_rows]
    by_state = defaultdict(list)
    for i, row in enumerate(rows):
        if row.state:
            by_state[row.state].append(i)
    packable = sum(1 for g in by_state.values() if len(g) >= 2)
    print(f"states with several questions: {packable}", flush=True)
    teacher_table = teaching.load(args.teacher_file) if args.teacher_file else {}
    if teacher_table:
        labelled = teaching.attach(rows, teacher_table)
        print(f"teacher labels on {len(labelled)} of {len(rows)} rows", flush=True)
    print(f"train {len(rows)}  dev {len(dev)}  device {device}", flush=True)

    tokenizer = build_tokenizer(args.backbone)
    model = SpanScorer(args.backbone, tokenizer,
                       gradient_checkpointing=args.grad_checkpoint).to(device)

    if args.init_backbone_from:
        reused, total = reuse_encoder(model.model, args.init_backbone_from)
        print(f"reused encoder weights: {reused} of {total} tensors", flush=True)
        if reused == 0:
            sys.exit("no encoder weight matched; the checkpoint layout is not "
                     "what this expects, refusing to train from a cold backbone")

    teacher = None
    if args.teacher:
        from gavel import Gavel

        teacher = Gavel.from_checkpoint(args.teacher, device, args.max_length)
        print(f"teacher: {args.teacher}", flush=True)

    head = [p for n, p in model.named_parameters() if n.startswith("head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head")]
    groups = [{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}]
    if args.adam8bit:
        import bitsandbytes as bnb

        optimiser = bnb.optim.AdamW8bit(groups, weight_decay=0.01)
    else:
        optimiser = torch.optim.AdamW(groups, weight_decay=0.01,
                                      fused=device.startswith("cuda"))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[args.lr, args.head_lr], total_steps=args.steps,
        pct_start=args.warmup)

    best, history, step, started = -1.0, [], 0, time.time()
    stream = batches(rows, args.batch_size, rng, pack=args.pack, by_state=by_state,
                     pack_share=args.pack_share)
    model.train()
    while step < args.steps:
        try:
            prepared, targets, chunk, permutations = next(stream)
        except StopIteration:
            stream = batches(rows, args.batch_size, rng, pack=args.pack, by_state=by_state,
                             pack_share=args.pack_share)
            continue
        step += 1
        batch = encode(tokenizer, prepared, args.max_length, device)
        target, questions = gold_slots(batch, targets, device)
        # bf16 autocast, as the pair model was trained, calibrated and
        # measured. The first version ran everything in fp32, which on a
        # 3090 is half the tensor-core rate for nothing.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.startswith("cuda")):
            scores = model(batch)
        log_probabilities = model.group_log_softmax(scores, batch.group)
        width = log_probabilities.size(-1)

        soft = (teacher_distribution(teacher, prepared, device, args.max_length, width)
                if teacher is not None else None)
        if teacher_table:
            # Questions that Jev labelled take Jev's distribution; the
            # others keep the pair model's, if there is one.
            file_soft, has = file_distribution(chunk, permutations, width, device)
            soft = torch.where(has, file_soft, soft) if soft is not None else \
                (file_soft if has.any() else None)
        loss = packed_loss(log_probabilities, target, questions, batch.group >= 0, soft,
                           args.brier_weight, args.teacher_weight)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        schedule.step()

        if step % 100 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  f"{(time.time() - started) / step:.2f} s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            mean, per_source, flip = evaluate(model, tokenizer, dev, device,
                                              args.max_length, args.batch_size,
                                              args.flip_check)
            print(f"  dev mean over {len(per_source)} strata {mean:.4f}  "
                  f"option-order flip rate {flip:.4f}", flush=True)
            history.append({"step": step, "mean": mean, "flip_rate": flip,
                            "per_source": per_source})
            Path(args.out, "history.json").write_text(json.dumps(history, indent=2))
            if mean > best:
                best = mean
                torch.save({"model": model.state_dict(), "backbone": args.backbone,
                            "step": step, "mean": mean, "flip_rate": flip},
                           Path(args.out, "best.pt"))
                print(f"  saved best {best:.4f}", flush=True)

    print(f"best mean over strata {best:.4f}")


if __name__ == "__main__":
    main()
