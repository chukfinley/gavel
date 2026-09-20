#!/usr/bin/env python3
"""Ten languages, the same decision shapes.

The English-trained model scored 0.653 on English reading comprehension and
0.305 on the German rows of the same benchmark, which is chance. The skill does
not cross a language by itself, thus each language needs data.

Languages: English, German, French, Spanish, Portuguese, Italian, Russian,
Chinese, Japanese, Arabic and Hindi, as far as each source provides them.

Sources keep their original shape:
* XNLI — the inference anchor in fifteen languages.
* PAWS-X — "do these two sentences mean the same" as a binary decision.
* MASSIVE — intent routing with sampled option sets.
* Belebele — reading comprehension, four options, one passage.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl  # noqa: E402

NLI_OPTIONS = [
    Option("supported", "The evidence establishes the claim"),
    Option("insufficient", "The evidence does not establish either"),
    Option("contradicted", "The evidence establishes the opposite"),
]
SAME = [Option("yes", "Yes, they mean the same"), Option("no", "No, they differ in meaning")]

XNLI_LANGUAGES = ["en", "de", "fr", "es", "ru", "zh", "ar", "hi", "vi", "tr"]
PAWSX_LANGUAGES = ["en", "de", "fr", "es", "ja", "zh", "ko"]
MASSIVE_LANGUAGES = ["en", "de", "fr", "es", "pt", "it", "ru", "zh-CN", "ja", "ar", "hi"]
BELEBELE = {"eng_Latn": "en", "deu_Latn": "de", "fra_Latn": "fr", "spa_Latn": "es",
            "por_Latn": "pt", "ita_Latn": "it", "rus_Cyrl": "ru", "zho_Hans": "zh",
            "jpn_Jpan": "ja", "arb_Arab": "ar", "hin_Deva": "hi"}
# Two languages stay out of training, thus transfer can be measured.
HELD_OUT = {"ita_Latn", "hin_Deva"}


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/multilingual.jsonl")
    parser.add_argument("--out-test", default="data/test_multilingual.jsonl")
    parser.add_argument("--per-language", type=int, default=4000)
    parser.add_argument("--test-per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    train: list[Decision] = []
    test: list[Decision] = []

    def note(name: str, rows: list[Decision], where: list[Decision]) -> None:
        where.extend(rows)
        print(f"  {name}: {len(rows)}", flush=True)

    for language in XNLI_LANGUAGES:
        try:
            data = load_dataset("facebook/xnli", language, split="train").shuffle(seed=args.seed)
            rows = []
            for index, row in enumerate(data):
                if len(rows) >= args.per_language:
                    break
                if row["label"] not in (0, 1, 2):
                    continue
                rows.append(Decision(
                    id=rid("xnli", language, str(index)), state=row["premise"].strip(),
                    question=f"Assess the claim: {row['hypothesis'].strip()}",
                    options=list(NLI_OPTIONS), label=row["label"],
                    source=f"xnli-{language}", task="choice"))
            note(f"xnli-{language}", rows, train)
            held = load_dataset("facebook/xnli", language, split="test")
            rows = [Decision(id=rid("xnli-test", language, str(i)), state=r["premise"].strip(),
                             question=f"Assess the claim: {r['hypothesis'].strip()}",
                             options=list(NLI_OPTIONS), label=r["label"],
                             source=f"xnli-test-{language}", task="choice")
                    for i, r in enumerate(held) if r["label"] in (0, 1, 2)][: args.test_per_language]
            note(f"xnli-test-{language}", rows, test)
        except Exception as error:                               # noqa: BLE001
            print(f"  xnli-{language} failed: {error}", flush=True)

    for language in PAWSX_LANGUAGES:
        try:
            data = load_dataset("google-research-datasets/paws-x", language,
                                split="train").shuffle(seed=args.seed)
            rows = []
            for index, row in enumerate(data):
                if len(rows) >= args.per_language // 2:
                    break
                rows.append(Decision(
                    id=rid("pawsx", language, str(index)),
                    state=f"{row['sentence1'].strip()}\n{row['sentence2'].strip()}",
                    question="Do these two sentences mean the same thing?",
                    options=list(SAME), label=0 if int(row["label"]) == 1 else 1,
                    source=f"pawsx-{language}", task="noul"))
            note(f"pawsx-{language}", rows, train)
        except Exception as error:                               # noqa: BLE001
            print(f"  pawsx-{language} failed: {error}", flush=True)

    for language in MASSIVE_LANGUAGES:
        try:
            data = load_dataset("mteb/amazon_massive_intent", language, split="train")
            names = sorted({row["label_text"] for row in data})
            index_of = {name: i for i, name in enumerate(names)}
            rows = []
            for index, row in enumerate(data):
                if len(rows) >= args.per_language // 2:
                    break
                truth = index_of[row["label_text"]]
                others = [i for i in range(len(names)) if i != truth]
                picked = rng.sample(others, 5) + [truth]
                rng.shuffle(picked)
                rows.append(Decision(
                    id=rid("massive", language, str(index)), state=row["text"].strip(),
                    question="Which category does this request belong to?",
                    options=[Option(f"c{i}", names[i].replace("_", " ")) for i in picked],
                    label=picked.index(truth), source=f"massive-{language}", task="choice"))
            note(f"massive-{language}", rows, train)
        except Exception as error:                               # noqa: BLE001
            print(f"  massive-{language} failed: {error}", flush=True)

    for code, language in BELEBELE.items():
        try:
            data = load_dataset("facebook/belebele", code, split="test")
            rows = []
            for index, row in enumerate(data):
                target = int(row["correct_answer_num"]) - 1
                if target not in (0, 1, 2, 3):
                    continue
                answers = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
                rows.append(Decision(
                    id=rid("belebele", code, str(index)), state=row["flores_passage"].strip(),
                    question=row["question"].strip(),
                    options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(answers)],
                    label=target, source=f"belebele-{language}", task="choice"))
            rng.shuffle(rows)
            note(f"belebele-test-{language}", rows[: args.test_per_language], test)
            if code not in HELD_OUT:
                note(f"belebele-{language}", rows[args.test_per_language :], train)
        except Exception as error:                               # noqa: BLE001
            print(f"  belebele-{code} failed: {error}", flush=True)

    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    print(f"\ntrain {len(train)}  test {len(test)}")


if __name__ == "__main__":
    main()
