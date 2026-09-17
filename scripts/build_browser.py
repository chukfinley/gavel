#!/usr/bin/env python3
"""Browser actions as typed decisions.

One step of a browser agent is a closed choice: of the elements on the page,
which one does the next action apply to. The value that gets typed into a field
is open text and stays with a language model; the element and the operation do
not, and they are the majority of the steps.

The state carries three parts, because a decision model has no memory:
the goal, what has already been done, and the elements now on the page. Without
the history the model clicks the same button forever.

Mind2Web marks the correct element of each step (`pos_candidates`) among the
distractors on the same page (`neg_candidates`), thus the label is human and
exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedec.schema import Decision, Option, write_jsonl  # noqa: E402

STOP = Option("done", "The goal is reached, stop here")
STUCK = Option("stuck", "None of these elements leads to the goal")


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def describe(candidate) -> str:
    """A short readable description of one page element."""
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except Exception:                                        # noqa: BLE001
            return re.sub(r"\s+", " ", candidate)[:160]
    if not isinstance(candidate, dict):
        return str(candidate)[:160]
    attributes = candidate.get("attributes")
    if isinstance(attributes, str):
        try:
            attributes = json.loads(attributes)
        except Exception:                                        # noqa: BLE001
            attributes = {}
    attributes = attributes or {}
    tag = candidate.get("tag") or attributes.get("role") or "element"
    text = (attributes.get("text") or attributes.get("aria_label")
            or attributes.get("title") or attributes.get("placeholder")
            or attributes.get("value") or "")
    text = re.sub(r"\s+", " ", str(text)).strip()
    return f"{tag}: {text}"[:160] if text else f"{tag}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/browser.jsonl")
    parser.add_argument("--out-test", default="data/test_browser.jsonl")
    parser.add_argument("--distractors", type=int, default=5)
    parser.add_argument("--test-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=151)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    rows: list[Decision] = []

    data = load_dataset("osunlp/Mind2Web", split="train")
    for index, task in enumerate(data):
        goal = str(task["confirmed_task"]).strip()
        reprs = list(task.get("action_reprs") or [])
        actions = list(task.get("actions") or [])
        for position, action in enumerate(actions):
            positives = action.get("pos_candidates") or []
            negatives = action.get("neg_candidates") or []
            if not positives or len(negatives) < 2:
                continue
            correct = describe(positives[0])
            picked = rng.sample(list(negatives), min(args.distractors, len(negatives)))
            options = [Option(f"e{i}", describe(c)) for i, c in enumerate(picked)]
            options.append(Option("target", correct))
            options.extend([STOP, STUCK])
            order = list(range(len(options)))
            rng.shuffle(order)
            shuffled = [options[i] for i in order]
            history = " ".join(f"{step + 1}. {text}" for step, text in
                               enumerate(reprs[:position]))[-1200:]
            operation = action.get("operation") or {}
            op = operation.get("op", "CLICK") if isinstance(operation, dict) else "CLICK"
            rows.append(Decision(
                id=rid("m2w", str(index), str(position)),
                state=(f"Goal: {goal}\n"
                       f"Done so far: {history or 'nothing yet'}\n"
                       f"Next operation: {op}"),
                question="Which element on the page does the next action apply to?",
                options=shuffled,
                label=[o.id for o in shuffled].index("target"),
                source="browser-mind2web", task="choice"))

    rng.shuffle(rows)
    write_jsonl(args.out_test, rows[: args.test_rows])
    write_jsonl(args.out_train, rows[args.test_rows :])
    print(f"train {max(len(rows) - args.test_rows, 0)}  test {min(args.test_rows, len(rows))}")


if __name__ == "__main__":
    main()
