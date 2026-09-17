#!/usr/bin/env python3
"""Train the typed decision model.

One forward pass gives one logit for each option. The loss mixes cross entropy
with the Brier score, thus the probabilities stay usable as a threshold.
Option order is shuffled in every batch, which removes the position bias that
the OpenJev baseline reports as a known failure.
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
from torch.utils.data import DataLoader

from typedec.encoding import encode                      # noqa: E402
from typedec.losses import DecisionLoss, balanced_accuracy, expected_calibration_error  # noqa: E402
from typedec.model import OptionScorer, build_tokenizer  # noqa: E402
from typedec.schema import read_jsonl                    # noqa: E402


def collate(tokenizer, max_length: int, shuffle_options: bool, rng: random.Random):
    def inner(rows):
        return encode(rows, tokenizer, max_length, shuffle_options, rng)
    return inner


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu())
        labels.append(batch.labels.cpu())
    width = max(p.size(1) for p in probabilities)
    probabilities = torch.cat([torch.nn.functional.pad(p, (0, width - p.size(1))) for p in probabilities])
    labels = torch.cat(labels)
    prediction = probabilities.argmax(dim=-1)
    return {
        "accuracy": (prediction == labels).float().mean().item(),
        "balanced_accuracy": balanced_accuracy(prediction, labels),
        "ece": expected_calibration_error(probabilities, labels),
        "rows": labels.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--dev", default="data/dev.jsonl")
    parser.add_argument("--out", default="runs/base")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--warmup", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(args.backbone)
    model = OptionScorer(args.backbone)
    model.resize(tokenizer)
    model.to(device)

    train_rows = list(read_jsonl(args.train))
    dev_rows = list(read_jsonl(args.dev))
    if args.limit:
        train_rows = train_rows[: args.limit]
    print(f"train {len(train_rows)}  dev {len(dev_rows)}", flush=True)

    rng = random.Random(args.seed)
    train_loader = DataLoader(
        train_rows, batch_size=args.batch_size, shuffle=True, num_workers=2,
        collate_fn=collate(tokenizer, args.max_length, True, rng), drop_last=True)
    dev_loader = DataLoader(
        dev_rows, batch_size=args.batch_size, shuffle=False, num_workers=2,
        collate_fn=collate(tokenizer, args.max_length, False, rng))

    steps = int(len(train_loader) * args.epochs) // args.accum
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = int(steps * args.warmup)

    def schedule(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, schedule)
    criterion = DecisionLoss(args.brier_weight)

    best, step, started = 0.0, 0, time.time()
    history = []
    model.train()
    done = False
    while not done:
        for index, batch in enumerate(train_loader):
            batch = batch.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            loss, parts = criterion(logits.float(), batch.labels, batch.option_mask)
            (loss / args.accum).backward()
            if (index + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    rate = step * args.batch_size * args.accum / (time.time() - started)
                    print(f"step {step}/{steps} loss {loss.item():.4f} ce {parts['ce']:.4f} "
                          f"brier {parts['brier']:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                          f"{rate:.0f} rows/s", flush=True)
                if step % args.eval_every == 0 or step == steps:
                    metrics = evaluate(model, dev_loader, device)
                    metrics["step"] = step
                    history.append(metrics)
                    print(f"  dev {metrics}", flush=True)
                    if metrics["balanced_accuracy"] > best:
                        best = metrics["balanced_accuracy"]
                        torch.save({"model": model.state_dict(), "backbone": args.backbone,
                                    "args": vars(args)}, out / "best.pt")
                        tokenizer.save_pretrained(out / "tokenizer")
                    model.train()
                if step >= steps:
                    done = True
                    break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best dev balanced accuracy {best:.4f}")


if __name__ == "__main__":
    main()
