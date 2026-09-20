#!/usr/bin/env python3
"""Convert public labelled datasets into the typed-decision format.

Every source gives the same record shape. The option texts are written into
each record, thus the model sees a different answer space in almost every
batch and cannot learn fixed classes.

The WANLI rows that OpenJev evaluates on come from the WANLI *test* split.
Only the train split is used here, thus the comparison stays clean.
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
# HuggingFace NLI label order is entailment, neutral, contradiction.
NLI_MAP = {0: 0, 1: 1, 2: 2}

YES_NO = [Option("yes", "Yes"), Option("no", "No")]


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


GOLD_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


def nli(name: str, rows, question_field: str, limit: int, source: str) -> list[Decision]:
    """WANLI carries a string in `gold`; the other sets carry an integer label."""
    out = []
    for index, row in enumerate(rows):
        if len(out) >= limit:
            break
        label = row["label"] if "label" in row else GOLD_MAP.get(row.get("gold", ""), -1)
        if label not in NLI_MAP:
            continue
        hypothesis = row[question_field].strip()
        premise = row["premise"].strip()
        if not hypothesis or not premise:
            continue
        out.append(Decision(
            id=rid(source, str(index)),
            state=premise,
            question=f"Assess the claim: {hypothesis}",
            options=list(NLI_OPTIONS),
            label=NLI_MAP[label],
            source=source,
            task="choice",
        ))
    return out


def boolq(rows, limit: int) -> list[Decision]:
    out = []
    for index, row in enumerate(rows):
        if len(out) >= limit:
            break
        out.append(Decision(
            id=rid("boolq", str(index)),
            state=row["passage"].strip(),
            question=row["question"].strip().rstrip("?") + "?",
            options=list(YES_NO),
            label=0 if row["answer"] else 1,
            source="boolq",
            task="noul",
        ))
    return out


def multiple_choice(rows, limit: int, source: str, state_key: str,
                    question_key: str, choices_key: str, label_key: str) -> list[Decision]:
    out = []
    for index, row in enumerate(rows):
        if len(out) >= limit:
            break
        choices = row[choices_key]
        texts = choices["text"] if isinstance(choices, dict) else list(choices)
        labels = choices.get("label") if isinstance(choices, dict) else None
        answer = row[label_key]
        if labels is not None:
            if answer not in labels:
                continue
            target = labels.index(answer)
        else:
            target = int(answer)
        if not texts or target >= len(texts):
            continue
        out.append(Decision(
            id=rid(source, str(index)),
            state=(row.get(state_key) or "").strip(),
            question=row[question_key].strip(),
            options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(texts)],
            label=target,
            source=source,
            task="choice",
        ))
    return out


def intent(rows, limit: int, source: str, names: list[str], text_key: str,
           label_key: str, options_per_row: int, rng: random.Random) -> list[Decision]:
    """Intent classification with a *sampled* option set for each row.

    This is the part that teaches run-time answer spaces: the same utterance
    appears with different candidate sets, thus the model must read the option
    text instead of memorising a class index.
    """
    out = []
    for index, row in enumerate(rows):
        if len(out) >= limit:
            break
        truth = int(row[label_key])
        others = [i for i in range(len(names)) if i != truth]
        if len(others) < options_per_row - 1:
            continue
        picked = rng.sample(others, options_per_row - 1) + [truth]
        rng.shuffle(picked)
        out.append(Decision(
            id=rid(source, str(index)),
            state=row[text_key].strip(),
            question="Which category does this request belong to?",
            options=[Option(f"c{i}", names[i].replace("_", " ")) for i in picked],
            label=picked.index(truth),
            source=source,
            task="choice",
        ))
    return out


def stars(rows, limit: int, source: str, text_key: str, label_key: str) -> list[Decision]:
    tiers = [Option(f"t{i}", d) for i, d in enumerate([
        "Very negative", "Negative", "Neutral", "Positive", "Very positive"])]
    out = []
    for index, row in enumerate(rows):
        if len(out) >= limit:
            break
        out.append(Decision(
            id=rid(source, str(index)),
            state=row[text_key].strip()[:2000],
            question="Rate the sentiment of this text.",
            options=list(tiers),
            label=int(row[label_key]),
            source=source,
            task="score",
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--per-source", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    train: list[Decision] = []
    report: list[str] = []

    def take(name: str, rows: list[Decision]) -> None:
        train.extend(rows)
        report.append(f"{name:24s} {len(rows):>7d}")
        print(f"  {name}: {len(rows)}", flush=True)

    print("loading sources", flush=True)
    try:
        mnli = load_dataset("nyu-mll/multi_nli", split="train")
        take("mnli", nli("mnli", mnli, "hypothesis", args.per_source, "mnli"))
    except Exception as error:
        print("  mnli failed:", error, flush=True)

    try:
        wanli = load_dataset("alisawuffles/WANLI", split="train")
        take("wanli-train", nli("wanli", wanli, "hypothesis", args.per_source, "wanli-train"))
    except Exception as error:
        print("  wanli failed:", error, flush=True)

    for config in ("plain_text",):
        try:
            anli = load_dataset("facebook/anli", split="train_r3")
            take("anli-r3", nli("anli", anli, "hypothesis", args.per_source, "anli-r3"))
        except Exception as error:
            print("  anli failed:", error, flush=True)

    try:
        bq = load_dataset("google/boolq", split="train")
        take("boolq", boolq(bq, args.per_source))
    except Exception as error:
        print("  boolq failed:", error, flush=True)

    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        take("arc", multiple_choice(arc, args.per_source, "arc", "", "question",
                                    "choices", "answerKey"))
    except Exception as error:
        print("  arc failed:", error, flush=True)

    try:
        csqa = load_dataset("tau/commonsense_qa", split="train")
        take("commonsenseqa", multiple_choice(csqa, args.per_source, "commonsenseqa", "",
                                              "question", "choices", "answerKey"))
    except Exception as error:
        print("  commonsenseqa failed:", error, flush=True)

    try:
        banking = load_dataset("mteb/banking77", split="train")
        names = sorted({row["label_text"] for row in banking})
        index_of = {name: i for i, name in enumerate(names)}
        banking = banking.map(lambda r: {"label": index_of[r["label_text"]]})
        take("banking77", intent(banking, args.per_source, "banking77", names,
                                 "text", "label", 6, rng))
    except Exception as error:
        print("  banking77 failed:", error, flush=True)

    try:
        yelp = load_dataset("Yelp/yelp_review_full", split="train")
        take("yelp", stars(yelp, args.per_source // 2, "yelp", "text", "label"))
    except Exception as error:
        print("  yelp failed:", error, flush=True)

    rng.shuffle(train)
    cut = max(1, int(len(train) * 0.02))
    write_jsonl(Path(args.out) / "dev.jsonl", train[:cut])
    write_jsonl(Path(args.out) / "train.jsonl", train[cut:])
    print(f"\ntrain {len(train) - cut}  dev {cut}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
