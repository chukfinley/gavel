#!/usr/bin/env python3
"""Training rows that carry retrieved evidence, exactly as inference will.

Retrieval lifted MMLU from 0.293 to 0.367 on a model that had never seen an
evidence block in training. The model has no reason to trust that block: for
every row it ever saw, the state was the whole story.

These rows fix that. Knowledge questions get the same BM25 passages put in
front of the state that the serving path would fetch. A share of them get
deliberately wrong passages, so the model learns that evidence can be useless
and that it should then fall back on the question instead of following the
distractor.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.retrieve import Bm25, ground                       # noqa: E402
from gavel.schema import read_jsonl, write_jsonl              # noqa: E402

# Sources where the answer has to be recalled, not read.
RECALL = {"mmlu-aux", "medmcqa", "arc", "commonsenseqa", "openbookqa", "sciq-recall",
          "truthfulqa", "logiqa", "arabic-mmlu", "exams-multilingual", "aqua"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train_v6.jsonl")
    parser.add_argument("--out", default="data/grounded.jsonl")
    parser.add_argument("--passages", type=int, default=120000)
    parser.add_argument("--rows", type=int, default=40000)
    parser.add_argument("--misleading", type=float, default=0.15,
                        help="share that gets evidence for a different question")
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    wiki = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    passages = []
    for row in wiki:
        text = str(row["text"])[:1200].strip()
        if len(text) > 200:
            passages.append(f"{row['title']}: {text}")
        if len(passages) >= args.passages:
            break
    print(f"index: {len(passages)} passages", flush=True)
    index = Bm25(passages)

    pool = [r for r in read_jsonl(args.train) if r.source in RECALL]
    rng.shuffle(pool)
    pool = pool[: args.rows]
    print(f"rows to ground: {len(pool)}", flush=True)

    out = []
    for position, decision in enumerate(pool):
        copied = copy.deepcopy(decision)
        if rng.random() < args.misleading:
            # Evidence fetched for a different question: present, plausible and
            # useless. Without these the model learns to trust any block.
            other = pool[rng.randrange(len(pool))]
            ground(copied, index, passages=2) if other is decision else ground(
                copy.deepcopy(other), index, passages=2)
            query = other.question + " " + " ".join(o.description for o in other.options)
            found = index.search(query, 2)
            if found:
                copied.state = ("Evidence:\n" + "\n\n".join(found)
                                + ("\n\n" + decision.state if decision.state else ""))
            copied.source = f"{decision.source}+misleading"
        else:
            ground(copied, index, passages=3)
            copied.source = f"{decision.source}+grounded"
        copied.id = f"g{position:07d}"
        out.append(copied)
        if position % 5000 == 0 and position:
            print(f"  {position}", flush=True)

    rng.shuffle(out)
    write_jsonl(args.out, out)
    from collections import Counter
    print(f"wrote {len(out)} rows")
    print(dict(Counter(r.source for r in out).most_common(8)))


if __name__ == "__main__":
    main()
