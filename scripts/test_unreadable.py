#!/usr/bin/env python3
"""Does confidence fall when the model cannot read the state at all?

This is the failure mode Laya measured and published: their English
checkpoint fed Khmer scored 0.000 accuracy at 0.952 confidence. Calibration
fitted on readable input says nothing about unreadable input, and an abstain
threshold that does not fire there is worthless exactly where it is needed —
a caller in production gets a confident wrong answer instead of an escalation.

We have never run it on ourselves. Three conditions over the same rows, the
same questions and the same options:

* **readable** — the state as it is
* **empty** — no state at all, the floor for "nothing to go on"
* **unreadable** — a state of the same rough length in a script the model has
  never been trained on (Amharic, Khmer, Georgian, Tamil, Devanagari; the
  Hindi rows are excluded from our training mix on purpose)

What the run has to show: accuracy falls towards chance in the last two, and
**confidence falls with it**. If confidence on unreadable input stays near
the readable level, the probability is not usable as a gate and we should
stop selling it as one.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import read_jsonl

SCRIPTS = {"amharic": "20231101.am", "khmer": "20231101.km",
           "georgian": "20231101.ka", "tamil": "20231101.ta",
           "devanagari": "20231101.hi"}


def foreign_text(rng, needed: int) -> dict[str, list[str]]:
    """Paragraphs in scripts the model was never trained on."""
    from datasets import load_dataset

    out: dict[str, list[str]] = {}
    for name, config in SCRIPTS.items():
        pieces = []
        stream = load_dataset("wikimedia/wikipedia", config, split="train",
                              streaming=True)
        for row in stream:
            text = " ".join(str(row["text"]).split())
            # Skip the navigation boilerplate that leads some of these dumps.
            if len(text) > 400 and "<div" not in text[:200]:
                pieces.append(text)
            if len(pieces) >= needed:
                break
        out[name] = pieces
        print(f"  {name}: {len(pieces)} paragraphs", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--test", default="data/test_general.jsonl")
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", default="results/unreadable.json")
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [r for r in read_jsonl(args.test) if r.label is not None and r.state]
    rng.shuffle(rows)
    rows = rows[: args.rows]
    print(f"{len(rows)} rows", flush=True)

    foreign = foreign_text(rng, max(40, args.rows // 4))

    from gavel import Gavel

    judge = (Gavel.from_checkpoint(args.checkpoint, None, args.max_length)
             if args.checkpoint else
             Gavel.from_pretrained(args.model, None, args.max_length))

    def run(name, state_of):
        correct, confidences = [], []
        for row in rows:
            options = [o.description for o in row.options]
            verdict = judge.decide(state_of(row), row.question, options)
            correct.append(int(options.index(verdict.option) == row.label))
            confidences.append(verdict.confidence)
        accuracy = sum(correct) / len(correct)
        confidence = statistics.mean(confidences)
        chance = statistics.mean(1 / len(r.options) for r in rows)
        print(f"  {name:22s} accuracy {accuracy:.3f}  mean confidence "
              f"{confidence:.3f}  (chance {chance:.3f})", flush=True)
        return {"accuracy": accuracy, "confidence": confidence, "chance": chance}

    report = {"readable": run("readable", lambda r: r.state),
              "empty": run("empty state", lambda _: "")}
    for name, pieces in foreign.items():
        if not pieces:
            continue
        report[name] = run(f"unreadable ({name})",
                           lambda r, p=pieces: rng.choice(p)[: max(200, len(r.state))])

    readable = report["readable"]
    worst = max((v["confidence"] for k, v in report.items()
                 if k not in ("readable", "empty")), default=0.0)
    drop = readable["confidence"] - worst
    print(f"\nreadable confidence {readable['confidence']:.3f}, highest "
          f"unreadable confidence {worst:.3f}, drop {drop:.3f}")
    if drop < 0.05:
        print("The gate does not fire on input the model cannot read. "
              "Do not sell the probability as an abstain signal.")
    else:
        print("Confidence falls on unreadable input, so the gate is worth "
              "something there.")
    report["confidence_drop"] = drop
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
