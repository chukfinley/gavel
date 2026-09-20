#!/usr/bin/env python3
"""Score the OpenJev fixtures with a trained typed decision model.

The fixtures already use this record format, thus they are read directly.
The numbers printed here are the same metrics OpenJev reports, therefore the
rows can be compared against their Qwen3.5-4B baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from gavel.encoding import encode_options, to_device   # noqa: E402
from gavel.losses import balanced_accuracy, expected_calibration_error  # noqa: E402
from gavel.model import EntailmentScorer, build_tokenizer  # noqa: E402
from gavel.schema import read_jsonl                    # noqa: E402

BASELINE = {                       # OpenJev results/phase1-summary.json
    "authored144": 0.8132381608,
    "wanli256": 0.6365253078,
    "typesafe102": 0.8453205128,
    "every36": 0.8055555556,
}


@torch.no_grad()
def score_file(model, tokenizer, path: Path, device, batch_size: int, max_length: int):
    rows = list(read_jsonl(path))
    probabilities, labels = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoding, mask, _ = encode_options(chunk, tokenizer, max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.option_logits(to_device(encoding, device), mask.to(device))
        probability = torch.softmax(logits.float(), dim=-1).cpu()
        width = probability.size(1)
        for row, single in zip(chunk, probability):
            probabilities.append(single[:width])
            labels.append(row.label)
    width = max(p.size(0) for p in probabilities)
    stacked = torch.stack([torch.nn.functional.pad(p, (0, width - p.size(0))) for p in probabilities])
    labels = torch.tensor(labels)
    prediction = stacked.argmax(dim=-1)
    return {
        "rows": len(rows),
        "accuracy": (prediction == labels).float().mean().item(),
        "balanced_accuracy": balanced_accuracy(prediction, labels),
        "ece": expected_calibration_error(stacked, labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixtures", nargs="+", required=True,
                        help="name=path pairs, e.g. authored144=../openjev/benchmarks/data/authored144.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = build_tokenizer(state["backbone"])
    model = EntailmentScorer(state["backbone"])
    model.load_state_dict(state["model"])
    model.to("cuda").eval()

    report = {}
    for pair in args.fixtures:
        name, _, path = pair.partition("=")
        metrics = score_file(model, tokenizer, Path(path), torch.device("cuda"),
                             args.batch_size, args.max_length)
        base = BASELINE.get(name)
        metrics["openjev_qwen3.5_4b"] = base
        if base is not None:
            metrics["delta"] = round(metrics["balanced_accuracy"] - base, 4)
        report[name] = metrics
        print(f"{name:14s} rows {metrics['rows']:>4d}  "
              f"bal.acc {metrics['balanced_accuracy']:.4f}  "
              f"acc {metrics['accuracy']:.4f}  ece {metrics['ece']:.4f}  "
              + (f"baseline {base:.4f}  delta {metrics['delta']:+.4f}" if base else ""))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
