#!/usr/bin/env python3
"""Score the general test set, one stratum for each source."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from typedec.encoding import encode                      # noqa: E402
from typedec.losses import balanced_accuracy, expected_calibration_error  # noqa: E402
from typedec.model import OptionScorer, build_tokenizer  # noqa: E402
from typedec.schema import read_jsonl                    # noqa: E402


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test", default="data/test_general.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = build_tokenizer(state["backbone"])
    model = OptionScorer(state["backbone"])
    model.resize(tokenizer)
    model.load_state_dict(state["model"])
    model.to("cuda").eval()

    groups: dict[str, list] = defaultdict(list)
    for row in read_jsonl(args.test):
        groups[row.source].append(row)

    report = {}
    for source, rows in sorted(groups.items()):
        probabilities, labels = [], []
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start : start + args.batch_size]
            batch = encode(chunk, tokenizer, args.max_length).to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            probabilities.append(torch.softmax(logits.float(), dim=-1).cpu())
            labels.extend(row.label for row in chunk)
        width = max(p.size(1) for p in probabilities)
        stacked = torch.cat([torch.nn.functional.pad(p, (0, width - p.size(1))) for p in probabilities])
        target = torch.tensor(labels)
        prediction = stacked.argmax(dim=-1)
        report[source] = {
            "rows": len(rows),
            "accuracy": (prediction == target).float().mean().item(),
            "balanced_accuracy": balanced_accuracy(prediction, target),
            "ece": expected_calibration_error(stacked, target),
        }
        print(f"{source:22s} rows {len(rows):>4d}  acc {report[source]['accuracy']:.4f}  "
              f"bal.acc {report[source]['balanced_accuracy']:.4f}  ece {report[source]['ece']:.4f}")

    mean = sum(v["balanced_accuracy"] for v in report.values()) / len(report)
    print(f"\nmean over {len(report)} strata: {mean:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"strata": report, "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
