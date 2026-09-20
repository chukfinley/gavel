#!/usr/bin/env python3
"""JevBench: the leaderboard the Jev-class systems actually compete on.

231 public items in three tiers — easy 48, standard 72 (the file is called
`original.jsonl`), hard 111 — from <https://github.com/fstandhartinger/jevbench>,
scored as argmax accuracy over the item's own label set.

Published per-tier accuracy on the same public items, read from decider's
README on 2026-09-20 (the leaderboard's own score also covers 303 held-out
items plus speed and cost, so this is a partial comparison):

    system                                    easy   standard  hard
    GPT-5.6 Luna, low reasoning               1.000  0.972     0.964
    Jev 1.13.0 (TypeSafe, closed)             1.000  0.986     0.730
    djev (Maisa, diffusion-gemma)             1.000  0.986     0.676
    OpenJev (DiffusionGemma 26B-A4B)          1.000  0.972     0.640
    SemIf (Qwen3.5-4B)                        1.000  0.986     0.613
    open-alternative-jev (Qwen3.5-4B)         1.000  0.833     0.568
    system-one-open (Gemma 4 E2B)             1.000  0.931     0.486
    system-one (Qwen3-8B)                     1.000  0.889     0.486
    decider-2b v10 (1.9 B)                    1.000  0.847     0.459
    Bespoke Nimble 9B                         1.000  0.931     0.369
    open-jev-deberta-v3-large (435 M encoder) 1.000  0.431     0.378

The last row is the one to beat first: it is the only encoder of our size
class on the board. The hard tier is where a long context should pay — its
`long_policy` family runs to 3746 tokens of state, which Von at 512 and
kotoba at 256 cannot read at all.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO = "https://github.com/fstandhartinger/jevbench"
TIERS = {"easy": "easy.jsonl", "standard": "original.jsonl", "hard": "hard.jsonl"}
PUBLISHED = {
    "gpt-5.6-luna": (1.000, 0.972, 0.964),
    "jev-1.13.0": (1.000, 0.986, 0.730),
    "djev": (1.000, 0.986, 0.676),
    "openjev-26b": (1.000, 0.972, 0.640),
    "semif-4b": (1.000, 0.986, 0.613),
    "decider-2b-v10": (1.000, 0.847, 0.459),
    "nimble-9b": (1.000, 0.931, 0.369),
    "open-jev-deberta": (1.000, 0.431, 0.378),
}


def options_for(item) -> tuple[list[str], list[str]]:
    """The label the gold uses, and the text the model reads for it.

    Two shapes, and getting this wrong is silent. A choice or a noul carries
    `criteria` as a mapping, keyed by the label, or by "true"/"false" where
    the labels are "yes"/"no". A **score carries a list**, one description
    per level in order, and the labels are the level numbers. Reading the
    list as a mapping is what hid every level description from the model in
    the other evaluator.
    """
    question = item["question"]
    criteria = question.get("criteria")
    labels = [str(label) for label in item["labels"]]
    if isinstance(criteria, list):
        return labels, [f"{label}: {criteria[index]}" if index < len(criteria)
                        else label for index, label in enumerate(labels)]
    criteria = criteria or {}
    alias = {"yes": "true", "no": "false"}
    texts = []
    for label in labels:
        description = criteria.get(label) or criteria.get(alias.get(label, label))
        texts.append(f"{label}: {description}" if description else label)
    return labels, texts


def state_text(item) -> str:
    """The hard tier sends JSON states, not only strings."""
    state = item["state"]
    return state if isinstance(state, str) else json.dumps(state, indent=1)


def load(folder: Path, tiers: list[str]) -> dict:
    if not folder.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", REPO, str(folder)],
                       check=True)
    out = {}
    for tier in tiers:
        path = folder / "datasets" / "public" / TIERS[tier]
        out[tier] = [json.loads(line) for line in path.read_text().splitlines() if line]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--span-checkpoint", default="",
                        help="a single-sequence checkpoint from train_span.py")
    parser.add_argument("--suite", default="_scratch/jevbench")
    parser.add_argument("--tiers", default="easy,standard,hard")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip() in TIERS]
    data = load(Path(args.suite), tiers)

    if args.span_checkpoint:
        from gavel.spanapi import SpanGavel

        judge = SpanGavel.from_checkpoint(args.span_checkpoint, args.device,
                                          args.max_length)
    else:
        from gavel import Gavel

        judge = (Gavel.from_checkpoint(args.checkpoint, args.device, args.max_length)
                 if args.checkpoint else
                 Gavel.from_pretrained(args.model, args.device, args.max_length))

    report, latencies = {}, []
    for tier in tiers:
        items = data[tier]
        hits = 0
        families: dict[str, list[int]] = defaultdict(list)
        for item in items:
            labels, texts = options_for(item)
            started = time.perf_counter()
            verdict = judge.decide(state_text(item),
                                   item["question"]["instructions"], texts)
            latencies.append((time.perf_counter() - started) * 1000)
            chosen = labels[texts.index(verdict.option)]
            correct = int(chosen == str(item["expected"]))
            hits += correct
            families[item["family"]].append(correct)
        accuracy = hits / len(items)
        report[tier] = {
            "accuracy": accuracy, "correct": hits, "items": len(items),
            "families": {name: {"accuracy": round(sum(v) / len(v), 4), "n": len(v)}
                         for name, v in sorted(families.items())}}
        print(f"\n=== {tier} ({len(items)} items) — {hits}/{len(items)} = {accuracy:.3f}")
        for name, scores in sorted(families.items()):
            print(f"    {name:18s} {sum(scores):3d}/{len(scores):<3d} "
                  f"= {sum(scores) / len(scores):.3f}")

    print("\n           " + "".join(f"{t:>10s}" for t in tiers))
    ours = "".join(f"{report[t]['accuracy']:10.3f}" for t in tiers)
    print(f"{'gavel':22s}{ours}")
    order = {"easy": 0, "standard": 1, "hard": 2}
    for name, values in PUBLISHED.items():
        row = "".join(f"{values[order[t]]:10.3f}" for t in tiers)
        print(f"{name:22s}{row}")

    report["latency_ms"] = statistics.mean(latencies) if latencies else 0.0
    report["published"] = PUBLISHED
    print(f"\nlatency {report['latency_ms']:.0f} ms per item on {judge.device}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
