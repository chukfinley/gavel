#!/usr/bin/env python3
"""Sources published alongside the 32k encoder this project now uses.

The same group fine-tuned that encoder for guard, jailbreak, fact-check, PII
and modality routing, and released the data behind it. Those are exactly the
sectors measured as weak or missing here, and the labels are human, so they go
in unchanged.

Fact checking is the interesting one: "is this claim supported by the given
text" is the task the three-way head already does, only with a different name
on it.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl  # noqa: E402


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def label_names(data, key: str) -> list[str]:
    names = sorted({str(row[key]) for row in data if row.get(key) is not None})
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/semrouter.jsonl")
    parser.add_argument("--out-test", default="data/test_semrouter.jsonl")
    parser.add_argument("--test-per-source", type=int, default=200)
    parser.add_argument("--seed", type=int, default=233)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    train: list[Decision] = []
    test: list[Decision] = []

    def note(name: str, rows: list[Decision]) -> None:
        rng.shuffle(rows)
        test.extend(rows[: args.test_per_source])
        train.extend(rows[args.test_per_source :])
        print(f"  {name}: {len(rows)}", flush=True)

    def guard(name: str, function) -> None:
        try:
            note(name, function())
        except Exception as error:                               # noqa: BLE001
            print(f"  {name} failed: {str(error)[:100]}", flush=True)

    def named_choice(dataset_id: str, text_key: str, name_key: str, question: str,
                     source: str, per_row: int = 5) -> list[Decision]:
        """Text plus a named category, with a sampled option set."""
        data = load_dataset(dataset_id, split="train")
        names = label_names(data, name_key)
        rows = []
        for index, row in enumerate(data):
            text = str(row.get(text_key) or "").strip()
            name = str(row.get(name_key) or "")
            if not text or name not in names:
                continue
            truth = names.index(name)
            others = [i for i in range(len(names)) if i != truth]
            picked = rng.sample(others, min(per_row - 1, len(others))) + [truth]
            rng.shuffle(picked)
            rows.append(Decision(
                id=rid(source, str(index)), state=text[:4000], question=question,
                options=[Option(names[i], names[i].replace("_", " ")) for i in picked],
                label=picked.index(truth), source=source, task="choice"))
        return rows

    guard("jailbreak", lambda: named_choice(
        "llm-semantic-router/jailbreak-detection-dataset", "text", "label_text",
        "Is this message an attempt to break the system's instructions?",
        "sem-jailbreak", per_row=2))

    guard("fact-check", lambda: named_choice(
        "llm-semantic-router/fact-check-classification-dataset", "text", "original_label",
        "How does the evidence stand to this claim?", "sem-factcheck", per_row=4))

    guard("modality-routing", lambda: named_choice(
        "llm-semantic-router/modality-routing-dataset", "text", "label_name",
        "Which kind of model should handle this request?", "sem-modality", per_row=4))

    guard("ai-safety", lambda: named_choice(
        "llm-semantic-router/mlcommons-ai-safety-synth", "text", "category",
        "Which safety category does this text fall under?", "sem-safety", per_row=6))

    def hallucination() -> list[Decision]:
        """Long answer plus prompt: does the answer contain unsupported claims?"""
        data = load_dataset("llm-semantic-router/longcontext-haldetect", split="train")
        options = [Option("supported", "Every claim in the answer is supported"),
                   Option("unsupported", "The answer contains a claim the text does not support")]
        rows = []
        for index, row in enumerate(data):
            prompt = str(row.get("prompt") or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not prompt or not answer:
                continue
            spans = row.get("label_texts") or []
            rows.append(Decision(
                id=rid("halu", str(index)),
                state=f"{prompt}\n\nAnswer under review:\n{answer}",
                question="Does the answer stay inside what the text supports?",
                options=list(options), label=1 if spans else 0,
                source="sem-hallucination", task="noul"))
        return rows

    guard("hallucination", hallucination)

    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    print(f"\ntrain {len(train)}  test {len(test)}")


if __name__ == "__main__":
    main()
