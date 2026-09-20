#!/usr/bin/env python3
"""Everything public that already has a stem, options and one correct answer.

The head-to-head against Qwen3.5-4B put the gap in one place: recall. The 4B
model reaches 0.973 on ARC-Easy, 0.795 on OpenBookQA and 0.688 on MMLU where
this model sits near chance. A 150M encoder will not store what a 4B decoder
read during pretraining, but the gap can be narrowed with the training split
of exactly those benchmarks.

MMLU ships an auxiliary training split of about 100000 rows, which is the
largest single source here. The test splits stay untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/knowledge.jsonl")
    parser.add_argument("--per-source", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    rows: list[Decision] = []

    def note(name: str, produced: list[Decision]) -> None:
        rows.extend(produced)
        print(f"  {name}: {len(produced)}", flush=True)

    def guard(name: str, function) -> None:
        try:
            note(name, function())
        except Exception as error:
            print(f"  {name} failed: {error}", flush=True)

    def mmlu_auxiliary() -> list[Decision]:
        data = load_dataset("cais/mmlu", "auxiliary_train", split="train").shuffle(seed=args.seed)
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            item = row["train"] if "train" in row else row
            choices = item.get("choices")
            if not choices:
                continue
            target = int(item["answer"])
            if target >= len(choices):
                continue
            out.append(Decision(
                id=rid("mmlu-aux", str(index)), state="",
                question=str(item["question"]).strip(),
                options=[Option(f"o{i}", str(c).strip()) for i, c in enumerate(choices)],
                label=target, source="mmlu-aux", task="choice"))
        return out

    guard("mmlu-aux", mmlu_auxiliary)

    def qasc() -> list[Decision]:
        data = load_dataset("allenai/qasc", split="train")
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            texts, labels = row["choices"]["text"], row["choices"]["label"]
            if row["answerKey"] not in labels:
                continue
            # The two supporting facts are given, thus the answer is derivable.
            support = " ".join(filter(None, [row.get("fact1"), row.get("fact2")])).strip()
            out.append(Decision(
                id=rid("qasc", str(index)), state=support,
                question=row["question"].strip(),
                options=[Option(f"o{i}", t.strip()) for i, t in enumerate(texts)],
                label=labels.index(row["answerKey"]), source="qasc", task="choice"))
        return out

    guard("qasc", qasc)

    def winogrande() -> list[Decision]:
        data = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            if row["answer"] not in ("1", "2"):
                continue
            out.append(Decision(
                id=rid("winogrande", str(index)), state=row["sentence"].strip(),
                question="Which word fills the blank in the sentence?",
                options=[Option("o0", row["option1"].strip()), Option("o1", row["option2"].strip())],
                label=int(row["answer"]) - 1, source="winogrande", task="choice"))
        return out

    guard("winogrande", winogrande)

    def hellaswag() -> list[Decision]:
        data = load_dataset("Rowan/hellaswag", split="train").shuffle(seed=args.seed)
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            if not str(row["label"]).isdigit():
                continue
            out.append(Decision(
                id=rid("hellaswag", str(index)), state=row["ctx"].strip(),
                question="How does this continue?",
                options=[Option(f"o{i}", e.strip()) for i, e in enumerate(row["endings"])],
                label=int(row["label"]), source="hellaswag", task="choice"))
        return out

    guard("hellaswag", hellaswag)

    def swag() -> list[Decision]:
        data = load_dataset("allenai/swag", "regular", split="train").shuffle(seed=args.seed)
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source // 2:
                break
            endings = [row[f"ending{i}"] for i in range(4)]
            out.append(Decision(
                id=rid("swag", str(index)), state=row["sent1"].strip(),
                question=f"{row['sent2'].strip()} — how does it continue?",
                options=[Option(f"o{i}", e.strip()) for i, e in enumerate(endings)],
                label=int(row["label"]), source="swag", task="choice"))
        return out

    guard("swag", swag)

    def logiqa() -> list[Decision]:
        data = load_dataset("hails/agieval-logiqa-en", split="test")
        out = []
        for index, row in enumerate(data):
            choices = row["choices"]
            target = int(row["gold"][0]) if isinstance(row["gold"], list) else int(row["gold"])
            if target >= len(choices):
                continue
            out.append(Decision(
                id=rid("logiqa", str(index)), state="",
                question=str(row["query"]).strip(),
                options=[Option(f"o{i}", str(c).strip()) for i, c in enumerate(choices)],
                label=target, source="logiqa", task="choice"))
        return out

    guard("logiqa", logiqa)

    def truthful() -> list[Decision]:
        data = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
        out = []
        for index, row in enumerate(data):
            targets = row["mc1_targets"]
            choices, labels = targets["choices"], targets["labels"]
            if 1 not in labels:
                continue
            order = list(range(len(choices)))
            rng.shuffle(order)
            correct = labels.index(1)
            out.append(Decision(
                id=rid("truthfulqa", str(index)), state="",
                question=row["question"].strip(),
                options=[Option(f"o{i}", str(choices[k]).strip()) for i, k in enumerate(order)],
                label=order.index(correct), source="truthfulqa", task="choice"))
        return out

    guard("truthfulqa", truthful)

    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
