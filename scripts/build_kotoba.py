#!/usr/bin/env python3
"""The corpus published with open-jev-deberta, flattened into our record shape.

Their repository (kotoba-lang/typed-decisions, MIT) commits its built corpus:
one state with several typed questions each. Two things are worth taking:

* sources this project does not have — RTE, QNLI, MRPC, CoLA, emotion,
* their question wording, which differs from ours and therefore widens the
  range of phrasings the model has to read rather than recognise.

A state with N questions becomes N decisions here, because this model answers
one question per pass. Their one-pass-many-questions head is the part that is
not taken; that is an architecture change, not a data import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedec.schema import Decision, Option, write_jsonl  # noqa: E402

BASE = "https://raw.githubusercontent.com/kotoba-lang/typed-decisions/main"
FILES = [("data-multi/train.jsonl", "train"), ("data-fam/train.jsonl", "train"),
         ("data-multi/test.jsonl", "test"), ("data-fam/ood-test.jsonl", "test")]
# Sources already covered here; only their extra ones and their wording are new.
KNOWN = {"banking77", "boolq", "sst5", "clinc", "ag_news", "dbpedia"}


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def fetch(path: str) -> list[dict]:
    with urllib.request.urlopen(f"{BASE}/{path}", timeout=300) as response:
        return [json.loads(line) for line in response.read().decode().splitlines() if line.strip()]


def convert(rows: list[dict], tag: str, keep_known: bool) -> list[Decision]:
    out = []
    for index, row in enumerate(rows):
        state = str(row.get("state") or "").strip()
        source = str(row.get("source") or "kotoba")
        if not state or (not keep_known and source in KNOWN):
            continue
        for number, question in enumerate(row.get("questions") or []):
            options = question.get("options") or []
            gold = question.get("gold")
            kind = str(question.get("kind") or "choice")
            if kind == "noul" and not options:
                options = ["Yes", "No"]
                gold = 0 if gold in (1, True, "yes") else 1
            if not isinstance(gold, int) or gold >= len(options) or len(options) < 2:
                continue
            out.append(Decision(
                id=rid(tag, source, str(index), str(number)), state=state[:6000],
                question=str(question.get("instructions") or "").strip(),
                options=[Option(f"o{i}", str(o).strip()) for i, o in enumerate(options)],
                label=gold, source=f"kotoba-{source}", task=kind))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/kotoba.jsonl")
    parser.add_argument("--out-test", default="data/test_kotoba.jsonl")
    parser.add_argument("--keep-known", action="store_true",
                        help="also import sources this project already has")
    parser.add_argument("--seed", type=int, default=277)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train: list[Decision] = []
    test: list[Decision] = []
    for path, kind in FILES:
        try:
            rows = convert(fetch(path), path, args.keep_known)
            (train if kind == "train" else test).extend(rows)
            print(f"  {path}: {len(rows)}", flush=True)
        except Exception as error:                               # noqa: BLE001
            print(f"  {path} failed: {str(error)[:90]}", flush=True)

    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    from collections import Counter
    print(f"\ntrain {len(train)}  test {len(test)}")
    print(dict(Counter(r.source for r in train).most_common(12)))


if __name__ == "__main__":
    main()
