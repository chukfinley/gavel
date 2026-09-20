#!/usr/bin/env python3
"""A stratified development set, for choosing checkpoints honestly.

Run 3 showed why this is needed. Pooled development accuracy rose from 0.76 to
0.80 while the WANLI fixture fell from 0.697 to 0.612 and the TypeSafe fixture
from 0.450 to 0.328. The development set was drawn from the training mix, thus
the largest sources decided the number, and the trade that lost language
understanding looked like progress.

This file holds equal strata from validation splits and from held-out slices
that the training set does not touch. The selection number is the mean over
strata, so no source can buy the decision with its size.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, read_jsonl, write_jsonl

NLI_OPTIONS = [
    Option("supported", "The evidence establishes the claim"),
    Option("insufficient", "The evidence does not establish either"),
    Option("contradicted", "The evidence establishes the opposite"),
]
YES_NO = [Option("yes", "Yes"), Option("no", "No")]
GOLD = {"entailment": 0, "neutral": 1, "contradiction": 2}


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dev_strat.jsonl")
    parser.add_argument("--per-stratum", type=int, default=200)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    rows: list[Decision] = []
    cap = args.per_stratum

    def add(name: str, produced: list[Decision]) -> None:
        rows.extend(produced[:cap])
        print(f"  {name}: {len(produced[:cap])}", flush=True)

    def guard(name: str, function) -> None:
        try:
            add(name, function())
        except Exception as error:
            print(f"  {name} failed: {error}", flush=True)

    def nli(source: str, dataset, key: str = "label") -> list[Decision]:
        out = []
        for index, row in enumerate(dataset):
            if len(out) >= cap:
                break
            raw = row.get(key, row.get("gold"))
            label = GOLD.get(raw, -1) if isinstance(raw, str) else raw
            if label not in (0, 1, 2):
                continue
            out.append(Decision(
                id=rid(source, str(index)), state=row["premise"].strip(),
                question=f"Assess the claim: {row['hypothesis'].strip()}",
                options=list(NLI_OPTIONS), label=label, source=source, task="choice"))
        return out

    guard("anli-r1-dev", lambda: nli("anli-r1-dev", load_dataset("facebook/anli", split="dev_r1")))
    guard("anli-r3-dev", lambda: nli("anli-r3-dev", load_dataset("facebook/anli", split="dev_r3")))
    guard("snli-dev", lambda: nli("snli-dev", load_dataset("stanfordnlp/snli", split="validation")))
    # WANLI: the fixture uses the test split, and training used the first 40000
    # train rows, thus a later slice of train is unseen by both.
    guard("wanli-held", lambda: nli(
        "wanli-held", load_dataset("alisawuffles/WANLI", split="train").select(range(45000, 47000)), key="gold"))

    def boolq_held() -> list[Decision]:
        # The test set takes the first 400 validation rows; this slice starts later.
        data = load_dataset("google/boolq", split="validation").select(range(400, 1200))
        return [Decision(id=rid("boolq-held", str(i)), state=row["passage"].strip(),
                         question=row["question"].strip().rstrip("?") + "?",
                         options=list(YES_NO), label=0 if row["answer"] else 1,
                         source="boolq-held", task="noul")
                for i, row in enumerate(data)]

    guard("boolq-held", boolq_held)

    def mc(source: str, dataset, stem: str, choices: str, answer: str) -> list[Decision]:
        out = []
        for index, row in enumerate(dataset):
            if len(out) >= cap:
                break
            block = row[choices]
            texts = block["text"] if isinstance(block, dict) else list(block)
            labels = block.get("label") if isinstance(block, dict) else None
            key = row[answer]
            if labels is not None:
                if key not in labels:
                    continue
                target = labels.index(key)
            else:
                target = int(key)
            if target >= len(texts):
                continue
            out.append(Decision(
                id=rid(source, str(index)), state="", question=row[stem].strip(),
                options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(texts)],
                label=target, source=source, task="choice"))
        return out

    guard("arc-easy-dev", lambda: mc("arc-easy-dev", load_dataset(
        "allenai/ai2_arc", "ARC-Easy", split="validation"), "question", "choices", "answerKey"))
    guard("openbookqa-dev", lambda: mc("openbookqa-dev", load_dataset(
        "allenai/openbookqa", "main", split="validation"), "question_stem", "choices", "answerKey"))

    def mmlu_dev() -> list[Decision]:
        data = load_dataset("cais/mmlu", "all", split="validation").shuffle(seed=args.seed)
        return [Decision(id=rid("mmlu-dev", str(i)), state="", question=row["question"].strip(),
                         options=[Option(f"o{k}", str(c).strip()) for k, c in enumerate(row["choices"])],
                         label=int(row["answer"]), source="mmlu-dev", task="choice")
                for i, row in enumerate(data)]

    guard("mmlu-dev", mmlu_dev)

    def clinc_dev() -> list[Decision]:
        data = load_dataset("clinc/clinc_oos", "small", split="validation")
        names = data.features["intent"].names
        out = []
        for index, row in enumerate(data):
            if len(out) >= cap:
                break
            truth = int(row["intent"])
            others = [i for i in range(len(names)) if i != truth]
            picked = rng.sample(others, 5) + [truth]
            rng.shuffle(picked)
            out.append(Decision(
                id=rid("clinc-dev", str(index)), state=row["text"].strip(),
                question="Which category does this request belong to?",
                options=[Option(f"c{i}", names[i].replace("_", " ")) for i in picked],
                label=picked.index(truth), source="clinc-dev", task="choice"))
        return out

    guard("clinc-dev", clinc_dev)

    # Held-out slices of the project's own generated data.
    for name, path in [("packet-held", "data/long.jsonl"), ("router-held", "data/router.jsonl"),
                       ("abstain-held", "data/abstain_short.jsonl")]:
        try:
            pool = list(read_jsonl(path))
            rng.shuffle(pool)
            picked = pool[:cap]
            for row in picked:
                row.source = name
            add(name, picked)
        except Exception as error:
            print(f"  {name} failed: {error}", flush=True)

    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source] = counts.get(row.source, 0) + 1
    print(f"\nwrote {len(rows)} rows in {len(counts)} strata")


if __name__ == "__main__":
    main()
