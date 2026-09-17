#!/usr/bin/env python3
"""Fit the temperature on held-out rows.

Accuracy and calibration are different things. As training converges the model
gets sharper and the expected calibration error grows: it is right more often
and overstates its confidence at the same time. One scalar fixes most of that.

Only the temperature is fitted; no weight changes. The argmax cannot move,
therefore accuracy stays exactly the same and only the probabilities improve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.nn import functional as F

from typedec.encoding import encode_options, to_device                  # noqa: E402
from typedec.losses import expected_calibration_error                   # noqa: E402
from typedec.model import EntailmentScorer, build_tokenizer             # noqa: E402
from typedec.schema import read_jsonl                                   # noqa: E402


@torch.no_grad()
def collect(model, rows, tokenizer, max_length, batch_size):
    logits, labels = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoding, mask, label = encode_options(chunk, tokenizer, max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model.option_logits(to_device(encoding, "cuda"), mask.to("cuda"))
        logits.append(scores.float().cpu())
        labels.append(label)
    width = max(item.size(1) for item in logits)
    padded = torch.cat([F.pad(item, (0, width - item.size(1)), value=torch.finfo(torch.float32).min)
                        for item in logits])
    return padded, torch.cat(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev", default="data/dev_v2.jsonl")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = build_tokenizer(state["backbone"])
    model = EntailmentScorer(state["backbone"])
    model.load_state_dict(state["model"])
    model.to("cuda").eval()

    rows = list(read_jsonl(args.dev))[: args.rows]
    logits, labels = collect(model, rows, tokenizer, args.max_length, args.batch_size)

    before = expected_calibration_error(torch.softmax(logits, dim=-1), labels)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimiser.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    temperature = float(log_temperature.exp())
    after = expected_calibration_error(torch.softmax(logits / temperature, dim=-1), labels)

    accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
    print(f"rows {len(rows)}  accuracy {accuracy:.4f} (unchanged by scaling)")
    print(f"temperature {temperature:.4f}   ece {before:.4f} -> {after:.4f}")

    state["model"]["log_temperature"] = torch.tensor([float(log_temperature)])
    target = args.out or args.checkpoint.replace(".pt", "-calibrated.pt")
    torch.save(state, target)
    print(f"written {target}")


if __name__ == "__main__":
    main()
