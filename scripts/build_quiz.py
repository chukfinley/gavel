#!/usr/bin/env python3
"""Public question banks: the native shape of this model.

A quiz question is already a typed decision — a stem, a few options, one
correct. These sets therefore need no reshaping, and they cover the weak spot
the stratified development set shows: knowledge questions, where the model sat
near chance (MMLU 0.36, OpenBookQA 0.325).

Two kinds are kept apart on purpose:

* Questions with a passage (SciQ support text, Belebele). The answer is in the
  text, thus the model can reason instead of recall. This is the shape that
  matters for routing and for business decisions.
* Questions without a passage (MedMCQA, SciQ without support, AQuA). Pure
  recall. These show what a small model cannot do, and that limit belongs in
  the report.

Belebele also gives a German split, which shows whether the skill survives a
change of language.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/quiz.jsonl")
    parser.add_argument("--out-test", default="data/test_quiz.jsonl")
    parser.add_argument("--per-source", type=int, default=12000)
    parser.add_argument("--test-per-source", type=int, default=400)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    train: list[Decision] = []
    test: list[Decision] = []

    def note(name: str, rows: list[Decision], where: list[Decision]) -> None:
        where.extend(rows)
        print(f"  {name}: {len(rows)}", flush=True)

    def guard(name: str, function) -> None:
        try:
            function()
        except Exception as error:                               # noqa: BLE001
            print(f"  {name} failed: {error}", flush=True)

    def sciq() -> None:
        def build(split: str, limit: int, source: str, with_support: bool) -> list[Decision]:
            data = load_dataset("allenai/sciq", split=split)
            out = []
            for index, row in enumerate(data):
                if len(out) >= limit:
                    break
                answers = [row["correct_answer"], row["distractor1"],
                           row["distractor2"], row["distractor3"]]
                order = list(range(4))
                rng.shuffle(order)
                support = (row.get("support") or "").strip()
                if with_support and not support:
                    continue
                out.append(Decision(
                    id=rid(source, str(index)),
                    state=support if with_support else "",
                    question=row["question"].strip(),
                    options=[Option(f"o{i}", str(answers[k]).strip()) for i, k in enumerate(order)],
                    label=order.index(0), source=source, task="choice"))
            return out

        note("sciq-with-passage", build("train", args.per_source, "sciq-passage", True), train)
        note("sciq-no-passage", build("train", args.per_source // 3, "sciq-recall", False), train)
        note("sciq-test", build("test", args.test_per_source, "sciq-test", True), test)

    guard("sciq", sciq)

    def medmcqa() -> None:
        data = load_dataset("openlifescienceai/medmcqa", split="train").shuffle(seed=args.seed)
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source:
                break
            answers = [row["opa"], row["opb"], row["opc"], row["opd"]]
            target = int(row["cop"])
            if target not in (0, 1, 2, 3):
                continue
            out.append(Decision(
                id=rid("medmcqa", str(index)), state="",
                question=row["question"].strip(),
                options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(answers)],
                label=target, source="medmcqa", task="choice"))
        note("medmcqa", out, train)
        validation = load_dataset("openlifescienceai/medmcqa", split="validation")
        held = []
        for index, row in enumerate(validation):
            if len(held) >= args.test_per_source:
                break
            answers = [row["opa"], row["opb"], row["opc"], row["opd"]]
            target = int(row["cop"])
            if target not in (0, 1, 2, 3):
                continue
            held.append(Decision(
                id=rid("medmcqa-test", str(index)), state="",
                question=row["question"].strip(),
                options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(answers)],
                label=target, source="medmcqa-test", task="choice"))
        note("medmcqa-test", held, test)

    guard("medmcqa", medmcqa)

    def belebele() -> None:
        for language, tag in [("eng_Latn", "belebele-en"), ("deu_Latn", "belebele-de")]:
            data = load_dataset("facebook/belebele", language, split="test")
            rows = []
            for index, row in enumerate(data):
                answers = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
                target = int(row["correct_answer_num"]) - 1
                if target not in (0, 1, 2, 3):
                    continue
                rows.append(Decision(
                    id=rid(tag, str(index)), state=row["flores_passage"].strip(),
                    question=row["question"].strip(),
                    options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(answers)],
                    label=target, source=tag, task="choice"))
            rng.shuffle(rows)
            # English feeds training and testing; German is kept for testing only,
            # thus it shows transfer to a language the training never used.
            if tag == "belebele-en":
                note(tag, rows[args.test_per_source : args.test_per_source + args.per_source], train)
            note(f"{tag}-test", rows[: args.test_per_source], test)

    guard("belebele", belebele)

    def aqua() -> None:
        data = load_dataset("deepmind/aqua_rat", "raw", split="train").shuffle(seed=args.seed)
        letters = ["A", "B", "C", "D", "E"]
        out = []
        for index, row in enumerate(data):
            if len(out) >= args.per_source // 3:
                break
            if row["correct"] not in letters:
                continue
            options = [str(o) for o in row["options"]]
            target = letters.index(row["correct"])
            if target >= len(options):
                continue
            out.append(Decision(
                id=rid("aqua", str(index)), state="",
                question=row["question"].strip(),
                options=[Option(f"o{i}", o.split(")", 1)[-1].strip()) for i, o in enumerate(options)],
                label=target, source="aqua", task="choice"))
        note("aqua", out, train)

    guard("aqua", aqua)

    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    print(f"\ntrain {len(train)}  test {len(test)}")


if __name__ == "__main__":
    main()
