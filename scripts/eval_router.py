#!/usr/bin/env python3
"""Router evaluation on held-out RouterBench prompts.

Accuracy alone does not say whether a router is worth using. What counts is
the quality it reaches and the money it spends. This script therefore reports,
on prompts the training never saw:

* tier accuracy — did it name the cheapest tier that solves the item,
* realised quality — of the items it routed, how many the chosen tier answers,
* spend — relative cost of the chosen tiers,

against four references: always cheap, always frontier, random, and an oracle
that always picks the cheapest tier that works.

The held-out rows are the prompts that `build_router.py` skipped because of its
per-benchmark cap. The same seed reproduces its selection, thus the complement
is provably unseen.

Cost units are relative, one unit = the small tier. They follow the public list
prices of the panel at the time RouterBench was collected and are assumptions,
not measurements; the ratios matter, not the absolute values.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from typedec.encoding import encode_options, to_device                  # noqa: E402
from typedec.model import EntailmentScorer, build_tokenizer             # noqa: E402
from typedec.schema import Decision, Option                             # noqa: E402

CHEAP = ["mistralai/mistral-7b-chat", "WizardLM/WizardLM-13B-V1.2"]
MIDDLE = ["mistralai/mixtral-8x7b-chat", "meta/llama-2-70b-chat",
          "gpt-3.5-turbo-1106", "claude-instant-v1"]
FRONTIER = ["gpt-4-1106-preview", "claude-v2", "claude-v1"]
COST = [1.0, 4.0, 90.0, 0.0]          # small, middle, frontier, human
TIERS = [
    Option("small", "A small fast chat model, lowest cost"),
    Option("middle", "A mid-size model, moderate cost"),
    Option("frontier", "A frontier model, highest cost and highest quality"),
    Option("human", "Send this to a person instead of a model"),
]


def solved(row, names) -> bool:
    return any(float(row[name]) >= 0.5 for name in names if name in row)


def tier_of(row) -> int:
    if solved(row, CHEAP):
        return 0
    if solved(row, MIDDLE):
        return 1
    if solved(row, FRONTIER):
        return 2
    return 3


def quality(row, tier: int) -> float:
    """Whether the chosen tier answers this item."""
    if tier == 0:
        return float(solved(row, CHEAP))
    if tier == 1:
        return float(solved(row, MIDDLE))
    if tier == 2:
        return float(solved(row, FRONTIER))
    return 1.0                                  # a person answers it


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--per-eval", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import pandas as pd
    from collections import defaultdict
    from huggingface_hub import hf_hub_download

    frame = pd.read_pickle(hf_hub_download(
        "withmartian/routerbench", "routerbench_0shot.pkl", repo_type="dataset"))

    # Reproduce the training selection to find what it consumed.
    rng = random.Random(args.seed)
    order = list(range(len(frame)))
    rng.shuffle(order)
    used, per_eval = set(), defaultdict(int)
    for position in order:
        row = frame.iloc[position]
        name = str(row["eval_name"])
        if per_eval[name] >= args.per_eval:
            continue
        prompt = str(row["prompt"]).strip()
        if not prompt or len(prompt) > 6000:
            continue
        per_eval[name] += 1
        used.add(position)

    held_out = [p for p in order if p not in used][: args.rows]
    print(f"held-out prompts: {len(held_out)} of {len(frame)}", flush=True)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = build_tokenizer(state["backbone"])
    model = EntailmentScorer(state["backbone"])
    model.load_state_dict(state["model"])
    model.to("cuda").eval()

    decisions, truth, records = [], [], []
    for position in held_out:
        row = frame.iloc[position]
        prompt = str(row["prompt"]).strip()
        if not prompt or len(prompt) > 6000:
            continue
        decisions.append(Decision(
            id=str(position), state=prompt,
            question="Which model should handle this request at the lowest cost that still answers it?",
            options=list(TIERS), label=0, source="router-test", task="choice"))
        truth.append(tier_of(row))
        records.append(row)

    predictions = []
    for start in range(0, len(decisions), args.batch_size):
        chunk = decisions[start : start + args.batch_size]
        encoding, mask, _ = encode_options(chunk, tokenizer, args.max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.option_logits(to_device(encoding, "cuda"), mask.to("cuda"))
        predictions.extend(logits.float().argmax(dim=-1).cpu().tolist())

    def summarise(name: str, chosen: list[int]) -> dict:
        got = [quality(row, tier) for row, tier in zip(records, chosen)]
        spend = [COST[tier] for tier in chosen]
        exact = sum(1 for tier, target in zip(chosen, truth) if tier == target) / len(chosen)
        result = {"tier_accuracy": round(exact, 4),
                  "quality": round(sum(got) / len(got), 4),
                  "relative_spend": round(sum(spend) / len(spend), 2),
                  "sent_to_person": round(sum(1 for t in chosen if t == 3) / len(chosen), 4)}
        print(f"{name:<18} tier-acc {result['tier_accuracy']:.4f}  "
              f"quality {result['quality']:.4f}  spend {result['relative_spend']:>6.2f}  "
              f"to person {result['sent_to_person']:.1%}", flush=True)
        return result

    report = {
        "typedec": summarise("typedec", predictions),
        "always_small": summarise("always small", [0] * len(records)),
        "always_middle": summarise("always middle", [1] * len(records)),
        "always_frontier": summarise("always frontier", [2] * len(records)),
        "random": summarise("random", [random.Random(1).randrange(3) for _ in records]),
        "oracle": summarise("oracle", truth),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
