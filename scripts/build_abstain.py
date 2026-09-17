#!/usr/bin/env python3
"""Teach the model to say "not determinable".

A decision model cannot invent text, thus it cannot hallucinate in the way a
chat model does. Its failure moves somewhere else: it picks a wrong option and
reports a high probability for it. OpenJev records exactly this on its
missing-evidence set — both systems answered above 0.8 confidence where the
honest answer was "insufficient".

Abstain rows are built, not guessed. Each one starts from a row whose answer
follows from one identified piece of the state. That piece is removed, and the
correct answer becomes "not determinable". The label is therefore exact and no
teacher model is involved.

Balance matters more than volume here. If abstain only ever appears on rows
where it is correct, the model learns "abstain is always right". Therefore the
option is also added to rows that stay decidable, where it must be rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedec.schema import Decision, Option, read_jsonl, write_jsonl  # noqa: E402

ABSTAIN = Option("not_determinable", "The information given does not decide this")


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def with_abstain(decision: Decision, rng: random.Random) -> Decision:
    """Same decision, with the abstain option added and still the old answer."""
    options = list(decision.options) + [ABSTAIN]
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    return Decision(
        id=rid("keep", decision.id), state=decision.state, question=decision.question,
        options=shuffled, label=order.index(decision.label),
        source=f"{decision.source}+abstain", task=decision.task)


def strip_packet(decision: Decision, rng: random.Random) -> Decision | None:
    """Remove the invoice line the question asks about."""
    try:
        packet = json.loads(decision.state)
        lines = packet["invoice"]["lines"]
    except Exception:                                            # noqa: BLE001
        return None
    number = None
    for token in decision.question.split():
        if token.isdigit():
            number = int(token)
            break
    if number is None:
        return None
    remaining = [line for line in lines if line["line"] != number]
    if len(remaining) == len(lines):
        return None
    packet["invoice"]["lines"] = remaining
    options = list(decision.options) + [ABSTAIN]
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    return Decision(
        id=rid("strip", decision.id), state=json.dumps(packet, indent=2),
        question=decision.question, options=shuffled,
        label=order.index(len(options) - 1),
        source="abstain-packet", task="choice")


def strip_incident(decision: Decision, rng: random.Random) -> Decision | None:
    """Remove the facts block that decides the severity."""
    try:
        packet = json.loads(decision.state)
    except Exception:                                            # noqa: BLE001
        return None
    if "facts" not in packet:
        return None
    packet.pop("facts")
    packet["note"] = "The structured facts for this incident were not attached to the report."
    options = list(decision.options) + [ABSTAIN]
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    return Decision(
        id=rid("strip-incident", decision.id), state=json.dumps(packet, indent=2),
        question=decision.question, options=shuffled,
        label=order.index(len(options) - 1),
        source="abstain-incident", task="choice")


def strip_state(decision: Decision, rng: random.Random) -> Decision:
    """Replace the decisive situation with a statement that it is missing."""
    replacements = [
        "The record for this case could not be retrieved.",
        "The attachment that describes this case is missing.",
        "The case was created without a description.",
    ]
    options = list(decision.options) + [ABSTAIN]
    order = list(range(len(options)))
    rng.shuffle(order)
    shuffled = [options[i] for i in order]
    return Decision(
        id=rid("blank", decision.id), state=rng.choice(replacements),
        question=decision.question, options=shuffled,
        label=order.index(len(options) - 1),
        source="abstain-missing", task="choice")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short", default="data/train_v2.jsonl")
    parser.add_argument("--long", default="data/long.jsonl")
    parser.add_argument("--out-short", default="data/abstain_short.jsonl")
    parser.add_argument("--out-long", default="data/abstain_long.jsonl")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="share of abstain-correct rows among the produced rows")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--long-count", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    short_rows = list(read_jsonl(args.short))
    produced: list[Decision] = []
    abstain_target = int(args.count * args.ratio)
    pool = [r for r in short_rows if r.source.startswith("synth-") or r.source in
            {"ag-news", "dbpedia", "banking77", "boolq"}]
    for decision in rng.sample(pool, min(abstain_target, len(pool))):
        produced.append(strip_state(decision, rng))
    for decision in rng.sample(short_rows, args.count - len(produced)):
        produced.append(with_abstain(decision, rng))
    rng.shuffle(produced)
    write_jsonl(args.out_short, produced)
    print(f"short: {len(produced)} rows "
          f"({sum(1 for r in produced if r.source.startswith('abstain'))} abstain-correct)")

    long_rows = list(read_jsonl(args.long))
    produced_long: list[Decision] = []
    packets = [r for r in long_rows if r.source == "synth-packet"]
    incidents = [r for r in long_rows if r.source == "synth-incident"]
    for decision in rng.sample(packets, min(args.long_count // 4, len(packets))):
        row = strip_packet(decision, rng)
        if row:
            produced_long.append(row)
    for decision in rng.sample(incidents, min(args.long_count // 4, len(incidents))):
        row = strip_incident(decision, rng)
        if row:
            produced_long.append(row)
    for decision in rng.sample(long_rows, max(args.long_count - len(produced_long), 0)):
        produced_long.append(with_abstain(decision, rng))
    rng.shuffle(produced_long)
    write_jsonl(args.out_long, produced_long)
    print(f"long: {len(produced_long)} rows "
          f"({sum(1 for r in produced_long if r.source.startswith('abstain'))} abstain-correct)")


if __name__ == "__main__":
    main()
