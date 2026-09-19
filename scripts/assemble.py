#!/usr/bin/env python3
"""Put the built sources together into the mixes the jobs train on.

This used to live inside the pod bootstrap as an inline block, which meant it
was never run on a developer machine and broke on the first rented pod. It is
a script now, so it can be tested like everything else.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedec.schema import read_jsonl, write_jsonl  # noqa: E402

TRAIN_PARTS = ["train.jsonl", "business.jsonl", "router.jsonl", "abstain_short.jsonl",
               "quiz.jsonl", "knowledge.jsonl", "multilingual.jsonl", "tools.jsonl",
               "browser.jsonl", "moderation.jsonl", "more.jsonl"]
LONG_PARTS = ["long.jsonl", "abstain_long.jsonl", "more_long.jsonl"]
AGENT_PARTS = ["tools.jsonl", "browser.jsonl"]
ROUTE = {"router-difficulty", "router-tier", "banking77", "ag-news", "dbpedia",
         "tweet-offensive", "tweet-hate", "sms-spam", "synth-routing", "synth-action",
         "mnli", "wanli-train", "anli-r3", "tickets-queue", "tickets-priority",
         "bitext-intent", "safety-aegis", "safety-injection"}
DOC = {"boolq", "quality", "race", "synth-packet", "synth-packet-long", "synth-incident",
       "synth-evidence", "sciq-passage", "mnli", "wanli-train", "anli-r3",
       "case-hold", "pubmedqa", "longbench-v2", "swiss-judgment"}
# Languages that stay out of training, so transfer can be measured.
SKIP_SUFFIX = ("-hi",)


def load(folder: Path, names: list[str]) -> list:
    rows = []
    for name in names:
        path = folder / name
        try:
            part = [r for r in read_jsonl(path) if not r.source.endswith(SKIP_SUFFIX)]
            rows += part
            print(f"  {name}: {len(part)}")
        except Exception as error:                               # noqa: BLE001
            print(f"  {name}: missing ({error})")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--dev-per-stratum", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    folder = Path(args.data)
    rng = random.Random(args.seed)

    print("training mix")
    rows = load(folder, TRAIN_PARTS)
    rng.shuffle(rows)
    write_jsonl(folder / "train_v6.jsonl", rows)

    print("long mix")
    long_rows = load(folder, LONG_PARTS)
    rng.shuffle(long_rows)
    write_jsonl(folder / "long_v2.jsonl", long_rows)

    print("agent mix")
    agent = load(folder, AGENT_PARTS)
    rng.shuffle(agent)
    write_jsonl(folder / "agent.jsonl", agent)

    write_jsonl(folder / "train_route.jsonl", [r for r in rows if r.source in ROUTE])
    write_jsonl(folder / "train_doc.jsonl",
                [r for r in rows if r.source in DOC or r.source.startswith("abstain")])

    # The selection set: equal strata from held-out material. Two language
    # strata move over from the multilingual test file so that checkpoint
    # selection can see whether the skill crosses a language.
    dev = list(read_jsonl(folder / "dev_strat.jsonl"))
    # The language slice is kept in its own file. Moving rows out of the test
    # file was destructive: a second assembly found nothing left to move and
    # silently produced a development set with two strata fewer, which makes
    # runs from different days incomparable.
    slice_path = folder / "dev_extra_multilingual.jsonl"
    try:
        if slice_path.exists():
            dev += list(read_jsonl(slice_path))
        else:
            test = list(read_jsonl(folder / "test_multilingual.jsonl"))
            move = {"xnli-test-fr", "belebele-ja"}
            extra = [r for r in test if r.source in move][: args.dev_per_stratum * 2]
            write_jsonl(slice_path, extra)
            write_jsonl(folder / "test_multilingual.jsonl",
                        [r for r in test if r.source not in move])
            dev += extra
    except Exception as error:                                   # noqa: BLE001
        print("  multilingual dev slice missing:", error)
    for name, tag in [("knowledge.jsonl", "knowledge-held"), ("moderation.jsonl", "moderation-held")]:
        try:
            pool = list(read_jsonl(folder / name))
            rng.shuffle(pool)
            for row in pool[: args.dev_per_stratum]:
                row.source = tag
            dev += pool[: args.dev_per_stratum]
        except Exception as error:                               # noqa: BLE001
            print(f"  {tag} missing:", error)
    rng.shuffle(dev)
    write_jsonl(folder / "dev_strat_v2.jsonl", dev)

    print(f"\ntrain_v6 {len(rows)}  long_v2 {len(long_rows)}  agent {len(agent)}  "
          f"dev_strat_v2 {len(dev)} in {len(Counter(r.source for r in dev))} strata")


if __name__ == "__main__":
    main()
