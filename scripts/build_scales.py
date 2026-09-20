#!/usr/bin/env python3
"""Ordered scales, in the shape a caller actually sends them.

The independent benchmark scores this model 0.333 on "how frustrated is this
customer, 0 to 2" and 0.444 on incident severity, while it scores 1.000 on
review sentiment. The difference is not difficulty: almost every training row
so far is a `choice`, and ordered levels appear only as review stars.

Two things are fixed here:

* **Levels get named criteria.** The benchmark sends
  `{"0": "calm and factual", "1": "annoyed", "2": "angry, threatening to leave"}`
  and the model has never seen an option written that way.
* **The scale is described in the question**, so that "0 to 2" and "low to
  critical" are the same task with different words, and neither is memorised.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, read_jsonl, write_jsonl  # noqa: E402

# The same underlying level, written the many ways a caller might write it.
WORDINGS = {
    3: [["0", "1", "2"], ["low", "medium", "high"], ["none", "some", "a lot"],
        ["calm", "annoyed", "angry"], ["minor", "moderate", "severe"]],
    4: [["0", "1", "2", "3"], ["low", "medium", "high", "critical"],
        ["trivial", "minor", "major", "blocking"]],
    5: [["1", "2", "3", "4", "5"], ["very low", "low", "medium", "high", "very high"],
        ["very negative", "negative", "neutral", "positive", "very positive"],
        ["one star", "two stars", "three stars", "four stars", "five stars"]],
}
CRITERIA = {
    "urgency": ["can wait, nothing is blocked", "should be looked at this week",
                "blocks work today", "the service is down for paying customers"],
    "frustration": ["calm and factual", "mildly annoyed", "clearly angry",
                    "threatening to leave"],
    "severity": ["cosmetic, one user", "some users, a workaround exists",
                 "many users, no workaround", "data exposed or the service is down"],
    "sentiment": ["strongly dislikes it", "dislikes it", "neither",
                  "likes it", "strongly likes it"],
}
QUESTIONS = [
    "Rate this on the scale below.",
    "Which level fits best?",
    "Pick the level that describes the text.",
    "On the given scale, where does this sit?",
]


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def rescale(label: int, source_levels: int, target_levels: int) -> int:
    """Map a level onto a scale with a different number of steps."""
    if source_levels <= 1:
        return 0
    position = label / (source_levels - 1)
    return min(target_levels - 1, round(position * (target_levels - 1)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="data/scales.jsonl")
    parser.add_argument("--rows", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=331)
    args = parser.parse_args()

    folder = Path(args.data)
    rng = random.Random(args.seed)
    pool: list[Decision] = []
    for name in ["train.jsonl", "business.jsonl", "more.jsonl", "moderation.jsonl",
                 "quiz.jsonl", "semrouter.jsonl"]:
        try:
            pool += [r for r in read_jsonl(folder / name)
                     if r.task == "score" or r.source in
                     ("yelp", "sst5-test", "tickets-priority", "synth-severity")]
        except Exception:                                        # noqa: BLE001
            continue
    rng.shuffle(pool)
    print(f"ordered rows available: {len(pool)}", flush=True)

    out = []
    for index, decision in enumerate(pool[: args.rows]):
        source_levels = len(decision.options)
        target_levels = rng.choice([k for k in WORDINGS if k <= max(source_levels, 3)])
        words = rng.choice(WORDINGS[target_levels])
        family = rng.choice(list(CRITERIA))
        descriptions = CRITERIA[family]
        options = []
        for position, word in enumerate(words):
            text = descriptions[min(position, len(descriptions) - 1)]
            # Half the rows carry the named criterion, half the bare level, so
            # that neither format is the only one the model has ever read.
            options.append(Option(word, f"{word}: {text}" if rng.random() < 0.5 else word))
        label = rescale(decision.label, source_levels, target_levels)
        scale = " to ".join([words[0], words[-1]])
        out.append(Decision(
            id=rid("scale", str(index)), state=decision.state,
            question=f"{rng.choice(QUESTIONS)} The scale runs from {scale}.",
            options=options, label=label, source="scales", task="score"))

    rng.shuffle(out)
    write_jsonl(args.out, out)
    print(f"wrote {len(out)} ordered-scale rows")


if __name__ == "__main__":
    main()
