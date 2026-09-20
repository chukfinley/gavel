#!/usr/bin/env python3
"""Export a checkpoint as a model anyone can load with transformers.

The training checkpoint holds the wrapper's state. What people want is the
plain sequence-classification model plus its tokenizer, so that

    AutoModelForSequenceClassification.from_pretrained("chukfinley/gavel-base")

works without this repository. The fitted temperature travels in the config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from gavel.model import EntailmentScorer, build_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = EntailmentScorer(state["backbone"])
    model.load_state_dict(state["model"])
    temperature = float(model.log_temperature.exp())

    inner = model.model
    inner.config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    inner.config.label2id = {v: k for k, v in inner.config.id2label.items()}
    inner.config.gavel = {
        "temperature": temperature,
        "decision_rule": "score each option as a hypothesis about the state, "
                         "take the entailment logit, softmax over the options",
        "premise": "the state, or the question when no state is given",
        "hypothesis": "'<question> The answer is <option>.', or "
                      "'<claim>. <option>.' for rows that assess a claim",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(out)
    build_tokenizer(state["backbone"]).save_pretrained(out)
    (out / "gavel_config.json").write_text(json.dumps(inner.config.gavel, indent=2))
    print(f"written {out}  temperature {temperature:.4f}  backbone {state['backbone']}")


if __name__ == "__main__":
    main()
