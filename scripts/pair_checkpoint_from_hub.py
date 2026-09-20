#!/usr/bin/env python3
"""Rebuild a pair-scorer checkpoint from a model published on the Hub.

The pod publishes the exported transformers model and keeps best.pt to
itself, so a second machine that wants to distil from a published model
needs the checkpoint layout back: EntailmentScorer keys under `model.` and
the fitted temperature as `log_temperature`. That is what this writes, and
`Gavel.from_checkpoint`, `train_span.py --teacher` and `--init-backbone-from`
all read it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--backbone", default="llm-semantic-router/Vela-1.0-Encoder-307M")
    parser.add_argument("--out", default="runs/pair_from_hub.pt")
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    temperature = float((getattr(config, "gavel", None) or {}).get("temperature", 1.0))
    model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=args.revision)
    state = {"model." + k: v for k, v in model.state_dict().items()}
    state["log_temperature"] = torch.tensor([math.log(temperature)])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state, "backbone": args.backbone,
                "source": f"{args.model}@{args.revision or 'main'}"}, args.out)
    print(f"wrote {args.out}: {len(state)} tensors, temperature {temperature:.4f}")


if __name__ == "__main__":
    main()
