#!/usr/bin/env python3
"""Router decisions with labels measured, not guessed.

RouterBench holds 36497 prompts and, for each of them, whether eleven models
answered correctly. That turns "how hard is this question" into a fact instead
of an opinion: a question is easy when the cheapest model already solves it,
and hard when only a frontier model does.

Two shapes come out of it, both with the option text supplied at run time:

* Difficulty, which a caller maps to a reasoning effort.
* Which class of model should take the request. This is the decision Vercel
  wires into its agent framework, where a decision model picks the language
  model for the turn.

A fourth answer exists in both shapes: no model in the panel solved the item.
Routing such a request to a bigger model only burns money, thus the honest
answer is to stop and ask a person.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedec.schema import Decision, Option, write_jsonl  # noqa: E402

CHEAP = ["mistralai/mistral-7b-chat", "WizardLM/WizardLM-13B-V1.2"]
MIDDLE = ["mistralai/mixtral-8x7b-chat", "meta/llama-2-70b-chat",
          "gpt-3.5-turbo-1106", "claude-instant-v1"]
FRONTIER = ["gpt-4-1106-preview", "claude-v2", "claude-v1"]

DIFFICULTY = [
    Option("easy", "Easy: a small model answers this correctly"),
    Option("medium", "Medium: it needs a mid-size model"),
    Option("hard", "Hard: it needs a frontier model"),
    Option("unsolved", "None of the available models answers this reliably"),
]

TIERS = [
    Option("small", "A small fast chat model, lowest cost"),
    Option("middle", "A mid-size model, moderate cost"),
    Option("frontier", "A frontier model, highest cost and highest quality"),
    Option("human", "Send this to a person instead of a model"),
]


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def tier_of(row, columns) -> int:
    def solved(names) -> bool:
        return any(float(row[name]) >= 0.5 for name in names if name in columns)

    if solved(CHEAP):
        return 0
    if solved(MIDDLE):
        return 1
    if solved(FRONTIER):
        return 2
    return 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/router.jsonl")
    parser.add_argument("--per-eval", type=int, default=1500,
                        help="cap for each benchmark, so that one does not dominate")
    parser.add_argument("--seed", type=int, default=53)
    args = parser.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    frame = pd.read_pickle(hf_hub_download(
        "withmartian/routerbench", "routerbench_0shot.pkl", repo_type="dataset"))
    columns = set(frame.columns)
    rng = random.Random(args.seed)

    per_eval: dict[str, int] = defaultdict(int)
    rows: list[Decision] = []
    counts: dict[str, int] = defaultdict(int)

    order = list(range(len(frame)))
    rng.shuffle(order)
    for position in order:
        row = frame.iloc[position]
        name = str(row["eval_name"])
        if per_eval[name] >= args.per_eval:
            continue
        prompt = str(row["prompt"]).strip()
        if not prompt or len(prompt) > 6000:
            continue
        per_eval[name] += 1
        label = tier_of(row, columns)
        counts[DIFFICULTY[label].id] += 1

        difficulty = list(DIFFICULTY)
        order_a = list(range(len(difficulty)))
        rng.shuffle(order_a)
        rows.append(Decision(
            id=rid("difficulty", str(position)), state=prompt,
            question="How hard is this request for a language model?",
            options=[difficulty[i] for i in order_a], label=order_a.index(label),
            source="router-difficulty", task="score",
            meta={"eval": name}))

        tiers = list(TIERS)
        order_b = list(range(len(tiers)))
        rng.shuffle(order_b)
        rows.append(Decision(
            id=rid("tier", str(position)), state=prompt,
            question="Which model should handle this request at the lowest cost that still answers it?",
            options=[tiers[i] for i in order_b], label=order_b.index(label),
            source="router-tier", task="choice",
            meta={"eval": name}))

    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    print("label share:", {key: f"{value / max(sum(counts.values()), 1):.1%}"
                           for key, value in sorted(counts.items())})
    print("benchmarks:", len(per_eval))


if __name__ == "__main__":
    main()
