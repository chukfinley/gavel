#!/usr/bin/env python3
"""Build a general test set from held-out splits of public datasets.

The OpenJev fixtures are small (144, 256, 102 and 36 rows). A difference of
three points on 102 rows means four rows. This file builds a wider test set so
that a result can be trusted.

Rules:
* Only test or validation splits, and only sources that the training set does
  not draw the same rows from.
* Each source keeps its own stratum. Accuracy is never pooled across sources,
  which follows the OpenJev metric contract.
* Three shapes are covered: binary (`noul`), multiple choice (`choice`) and
  ordered tiers (`score`).
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl

NLI_OPTIONS = [
    Option("supported", "The evidence establishes the claim"),
    Option("insufficient", "The evidence does not establish either"),
    Option("contradicted", "The evidence establishes the opposite"),
]
YES_NO = [Option("yes", "Yes"), Option("no", "No")]


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/test_general.jsonl")
    parser.add_argument("--per-source", type=int, default=400)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    rows: list[Decision] = []

    def add(name: str, produced: list[Decision]) -> None:
        rows.extend(produced)
        print(f"  {name}: {len(produced)}", flush=True)

    def guard(name: str, function) -> None:
        try:
            add(name, function())
        except Exception as error:
            print(f"  {name} failed: {error}", flush=True)

    def nli_rows(source: str, dataset, limit: int) -> list[Decision]:
        out = []
        for index, row in enumerate(dataset):
            if len(out) >= limit:
                break
            if row.get("label", -1) not in (0, 1, 2):
                continue
            out.append(Decision(
                id=rid(source, str(index)),
                state=row["premise"].strip(),
                question=f"Assess the claim: {row['hypothesis'].strip()}",
                options=list(NLI_OPTIONS), label=row["label"],
                source=source, task="choice"))
        return out

    guard("anli-r1-test", lambda: nli_rows(
        "anli-r1-test", load_dataset("facebook/anli", split="test_r1"), args.per_source))
    guard("anli-r2-test", lambda: nli_rows(
        "anli-r2-test", load_dataset("facebook/anli", split="test_r2"), args.per_source))
    guard("snli-test", lambda: nli_rows(
        "snli-test", load_dataset("stanfordnlp/snli", split="test"), args.per_source))

    def boolq_rows() -> list[Decision]:
        data = load_dataset("google/boolq", split="validation")
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            out.append(Decision(
                id=rid("boolq-val", str(index)), state=row["passage"].strip(),
                question=row["question"].strip().rstrip("?") + "?",
                options=list(YES_NO), label=0 if row["answer"] else 1,
                source="boolq-val", task="noul"))
        return out

    guard("boolq-val", boolq_rows)

    def arc_easy() -> list[Decision]:
        data = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            texts, labels = row["choices"]["text"], row["choices"]["label"]
            if row["answerKey"] not in labels:
                continue
            out.append(Decision(
                id=rid("arc-easy-test", str(index)), state="",
                question=row["question"].strip(),
                options=[Option(f"o{i}", t.strip()) for i, t in enumerate(texts)],
                label=labels.index(row["answerKey"]),
                source="arc-easy-test", task="choice"))
        return out

    guard("arc-easy-test", arc_easy)

    def openbookqa() -> list[Decision]:
        data = load_dataset("allenai/openbookqa", "main", split="test")
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            texts, labels = row["choices"]["text"], row["choices"]["label"]
            if row["answerKey"] not in labels:
                continue
            out.append(Decision(
                id=rid("openbookqa-test", str(index)), state="",
                question=row["question_stem"].strip(),
                options=[Option(f"o{i}", t.strip()) for i, t in enumerate(texts)],
                label=labels.index(row["answerKey"]),
                source="openbookqa-test", task="choice"))
        return out

    guard("openbookqa-test", openbookqa)

    def clinc() -> list[Decision]:
        data = load_dataset("clinc/clinc_oos", "small", split="test")
        names = data.features["intent"].names
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            truth = int(row["intent"])
            others = [i for i in range(len(names)) if i != truth]
            picked = rng.sample(others, 5) + [truth]
            rng.shuffle(picked)
            out.append(Decision(
                id=rid("clinc-test", str(index)), state=row["text"].strip(),
                question="Which category does this request belong to?",
                options=[Option(f"c{i}", names[i].replace("_", " ")) for i in picked],
                label=picked.index(truth), source="clinc-test", task="choice"))
        return out

    guard("clinc-test", clinc)

    def sst5() -> list[Decision]:
        data = load_dataset("SetFit/sst5", split="test")
        tiers = [Option(f"t{i}", d) for i, d in enumerate(
            ["Very negative", "Negative", "Neutral", "Positive", "Very positive"])]
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            out.append(Decision(
                id=rid("sst5-test", str(index)), state=row["text"].strip(),
                question="Rate the sentiment of this text.",
                options=list(tiers), label=int(row["label"]),
                source="sst5-test", task="score"))
        return out

    guard("sst5-test", sst5)

    def mmlu() -> list[Decision]:
        data = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=args.seed)
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            out.append(Decision(
                id=rid("mmlu-test", str(index)), state="",
                question=row["question"].strip(),
                options=[Option(f"o{i}", str(c).strip()) for i, c in enumerate(row["choices"])],
                label=int(row["answer"]), source="mmlu-test", task="choice"))
        return out

    guard("mmlu-test", mmlu)

    write_jsonl(args.out, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source] = counts.get(row.source, 0) + 1
    print(f"\nwrote {len(rows)} rows to {args.out}")
    for name, count in sorted(counts.items()):
        print(f"  {name:20s} {count}")


if __name__ == "__main__":
    main()
