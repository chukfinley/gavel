#!/usr/bin/env python3
"""Train the typed decision model.

Two objectives run together:

* **Anchor.** Natural language inference rows train the three-way head
  directly on (premise, hypothesis) pairs. This fixes the meaning of the
  entailment class; without it the decision objective has no ground to stand
  on and the model answers uniformly.
* **Decision.** Every row is scored option by option. The entailment logit of
  each option is its score, and a softmax over the options is the decision.
  The loss mixes cross entropy with the Brier score, thus the probability
  stays usable as a threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.nn import functional as F

from typedec.encoding import NLI_SOURCES, encode_anchor, encode_options, to_device  # noqa: E402
from typedec.losses import balanced_accuracy, brier, expected_calibration_error      # noqa: E402
from typedec.model import EntailmentScorer, build_tokenizer                          # noqa: E402
from typedec.schema import read_jsonl                                                # noqa: E402


@torch.no_grad()
def evaluate(model, rows, tokenizer, device, batch_size, max_length) -> dict:
    model.eval()
    probabilities, labels = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoding, mask, label = encode_options(chunk, tokenizer, max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.option_logits(to_device(encoding, device), mask.to(device))
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu())
        labels.append(label)
    width = max(p.size(1) for p in probabilities)
    stacked = torch.cat([F.pad(p, (0, width - p.size(1))) for p in probabilities])
    target = torch.cat(labels)
    prediction = stacked.argmax(dim=-1)
    model.train()
    return {
        "accuracy": (prediction == target).float().mean().item(),
        "balanced_accuracy": balanced_accuracy(prediction, target),
        "ece": expected_calibration_error(stacked, target),
        "rows": target.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--dev", default="data/dev.jsonl")
    parser.add_argument("--out", default="runs/base")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--decision-batch", type=int, default=4)
    parser.add_argument("--anchor-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--warmup", type=float, default=0.06)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-rows", type=int, default=768)
    parser.add_argument("--long", default="", help="jsonl with long-state decisions")
    parser.add_argument("--long-every", type=int, default=4, help="use a long batch every N steps")
    parser.add_argument("--long-batch", type=int, default=1)
    parser.add_argument("--long-max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args.backbone)
    model = EntailmentScorer(args.backbone).to(device)
    model.train()

    rows = list(read_jsonl(args.train))
    anchor_rows = [r for r in rows if r.source in NLI_SOURCES]
    long_rows = list(read_jsonl(args.long)) if args.long else []
    dev_rows = list(read_jsonl(args.dev))[: args.eval_rows]
    print(f"train {len(rows)}  anchor {len(anchor_rows)}  long {len(long_rows)}  "
          f"dev {len(dev_rows)}", flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = int(args.steps * args.warmup)

    def schedule(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(args.steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)

    best, started, history = 0.0, time.time(), []
    for step in range(1, args.steps + 1):
        anchor = rng.sample(anchor_rows, args.anchor_batch)
        encoding, labels = encode_anchor(anchor, tokenizer, args.max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            anchor_loss = F.cross_entropy(
                model.pair_logits(to_device(encoding, device)).float(), labels.to(device))
        anchor_loss.backward()

        # Every few steps the batch comes from the long pool, so that the model
        # learns to read states of a few thousand tokens. Training only on short
        # pairs gives a model that cannot use a long state at inference.
        use_long = long_rows and step % args.long_every == 0
        source_rows = long_rows if use_long else rows
        size = args.long_batch if use_long else args.decision_batch
        length = args.long_max_length if use_long else args.max_length
        decision = rng.sample(source_rows, size)
        encoding, mask, labels = encode_options(decision, tokenizer, length)
        mask, labels = mask.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.option_logits(to_device(encoding, device), mask).float()
        decision_loss = F.cross_entropy(logits, labels) + args.brier_weight * brier(logits, labels, mask)
        decision_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        optimiser.zero_grad(set_to_none=True)

        if step % 100 == 0:
            print(f"step {step}/{args.steps} anchor {anchor_loss.item():.4f} "
                  f"decision {decision_loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                  f"{step / (time.time() - started):.1f} steps/s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, dev_rows, tokenizer, device, 8, args.max_length)
            metrics["step"] = step
            history.append(metrics)
            print(f"  dev {metrics}", flush=True)
            if metrics["accuracy"] > best:
                best = metrics["accuracy"]
                torch.save({"model": model.state_dict(), "backbone": args.backbone,
                            "args": vars(args)}, out / "best.pt")
            (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"best dev accuracy {best:.4f}")


if __name__ == "__main__":
    main()
