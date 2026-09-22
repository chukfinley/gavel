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

from gavel.schema import read_jsonl, write_jsonl

TRAIN_PARTS = ["train.jsonl", "business.jsonl", "router.jsonl", "abstain_short.jsonl",
               "quiz.jsonl", "knowledge.jsonl", "multilingual.jsonl", "tools.jsonl",
               "browser.jsonl", "moderation.jsonl", "more.jsonl", "semrouter.jsonl", "kotoba.jsonl", "grounded.jsonl", "scales.jsonl",
               "criteria.jsonl", "claims.jsonl", "domains.jsonl", "games.jsonl", "routing.jsonl",
               "webnav.jsonl", "snake.jsonl", "jevdistill.jsonl", "emails_spam.jsonl",
               "products.jsonl"]
LONG_PARTS = ["long.jsonl", "abstain_long.jsonl", "more_long.jsonl"]
AGENT_PARTS = ["tools.jsonl", "browser.jsonl"]
ROUTE = {"router-difficulty", "router-tier", "banking77", "ag-news", "dbpedia",
         "tweet-offensive", "tweet-hate", "sms-spam", "synth-routing", "synth-action",
         "mnli", "wanli-train", "anli-r3", "tickets-queue", "tickets-priority",
         "bitext-intent", "safety-aegis", "safety-injection",
         "sem-jailbreak", "sem-modality", "sem-safety"}
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
        except Exception as error:
            print(f"  {name}: missing ({error})")
    return rows


def content_key(row) -> str:
    """The text of a decision, so a copy under another id is still found."""
    options = "\x1f".join(o.description.strip().lower() for o in row.options)
    return f"{row.state.strip().lower()}\x1e{row.question.strip().lower()}\x1e{options}"


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
    # The development strata are cut out of the mix further down and the rows
    # that go there are removed from training, so that checkpoint selection
    # reads held-out material. Before this, the "-held" strata were sampled
    # from files that stayed in the mix, which made those strata look better
    # than they were and pulled selection towards them.

    print("long mix")
    long_rows = load(folder, LONG_PARTS)
    rng.shuffle(long_rows)
    write_jsonl(folder / "long_v2.jsonl", long_rows)

    print("agent mix")
    agent = load(folder, AGENT_PARTS)
    rng.shuffle(agent)
    write_jsonl(folder / "agent.jsonl", agent)

    # The selection set: equal strata from held-out material. Two language
    # strata move over from the multilingual test file so that checkpoint
    # selection can see whether the skill crosses a language.
    held: set[str] = set()
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
    except Exception as error:
        print("  multilingual dev slice missing:", error)
    for name, tag in [("knowledge.jsonl", "knowledge-held"),
                      ("moderation.jsonl", "moderation-held"),
                      ("criteria.jsonl", "criteria-held"),
                      ("claims.jsonl", "claims-held"),
                      ("scales.jsonl", "scales-held"),
                      ("domains.jsonl", "domains-held")]:
        try:
            pool = list(read_jsonl(folder / name))
            rng.shuffle(pool)
            taken = pool[: args.dev_per_stratum]
            for row in taken:
                held.add(row.id)
                row.source = tag
            dev += taken
        except Exception as error:
            print(f"  {tag} missing:", error)
    rng.shuffle(dev)
    write_jsonl(folder / "dev_strat_v2.jsonl", dev)

    # Ids are not enough: the criteria and claims builders write several
    # wordings of one item under different ids, and a few sources repeat
    # an item verbatim. Anything with a dev row's text is held out too,
    # otherwise the dev score for that stratum measures recall of the
    # training set (1450 criteria rows leaked this way on 2026-09-21).
    held_text = {content_key(r) for r in dev}
    before = len(rows)
    rows = [r for r in rows if r.id not in held and content_key(r) not in held_text]
    print(f"\nheld out of training: {before - len(rows)} rows")
    write_jsonl(folder / "train_v6.jsonl", rows)
    write_jsonl(folder / "train_route.jsonl", [r for r in rows if r.source in ROUTE])
    write_jsonl(folder / "train_doc.jsonl",
                [r for r in rows if r.source in DOC or r.source.startswith("abstain")])

    print(f"train_v6 {len(rows)}  long_v2 {len(long_rows)}  agent {len(agent)}  "
          f"dev_strat_v2 {len(dev)} in {len(Counter(r.source for r in dev))} strata")


if __name__ == "__main__":
    main()
