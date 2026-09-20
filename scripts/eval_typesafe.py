#!/usr/bin/env python3
"""Score the TypeSafe fixture with the metric OpenJev uses.

Their number is not plain accuracy: rows are grouped by case (`group_id`), the
agreement is averaged inside each case, and then averaged over cases, so that a
case with many rows does not dominate. The second number is the total variation
distance to the published target distribution. Both are implemented here as in
`openjev/benchmarks/evaluate_external.py`, thus the result is comparable.

Published values for the 102 aligned rows:
    published Jev           modal agreement 0.8831   total variation 0.1268
    OpenJev direct logits   modal agreement 0.8453   total variation 0.1770
    OpenJev reranker        modal agreement 0.5595   total variation 0.4441
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from gavel.encoding import encode_options, to_device
from gavel.model import EntailmentScorer, build_tokenizer
from gavel.schema import from_json

PUBLISHED = {"jev": (0.8831410256, 0.1268250717), "openjev_direct": (0.8453205128, 0.1770346368)}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", default="/home/user/git/openjev/_fixtures/typesafe102.jsonl")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = build_tokenizer(state["backbone"])
    model = EntailmentScorer(state["backbone"])
    model.load_state_dict(state["model"])
    model.to("cuda").eval()

    raw = [json.loads(line) for line in open(args.fixture, encoding="utf-8") if line.strip()]
    cases = defaultdict(list)
    for row in raw:
        decision = from_json(row)
        encoding, mask, _ = encode_options([decision], tokenizer, args.max_length)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.option_logits(to_device(encoding, "cuda"), mask.to("cuda"))
        probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu().tolist()[: len(decision.options)]
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        target = row.get("target_distribution")
        variation = (sum(abs(a - b) for a, b in zip(probabilities, target)) / 2
                     if target else float("nan"))
        cases[row["group_id"]].append({"agreement": float(predicted == row["label"]),
                                       "total_variation": variation})

    agreement = statistics.mean(
        statistics.mean(item["agreement"] for item in rows) for rows in cases.values())
    variations = [statistics.mean(item["total_variation"] for item in rows)
                  for rows in cases.values() if not any(item["total_variation"] != item["total_variation"]
                                                        for item in rows)]
    variation = statistics.mean(variations) if variations else float("nan")

    print(f"rows {len(raw)}  cases {len(cases)}")
    print(f"equal-case modal agreement  {agreement:.4f}   "
          f"(OpenJev 4B {PUBLISHED['openjev_direct'][0]:.4f}, Jev {PUBLISHED['jev'][0]:.4f})")
    print(f"equal-case total variation  {variation:.4f}   "
          f"(OpenJev 4B {PUBLISHED['openjev_direct'][1]:.4f}, Jev {PUBLISHED['jev'][1]:.4f})")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"rows": len(raw), "cases": len(cases), "modal_agreement": agreement,
             "total_variation": variation, "published": PUBLISHED}, indent=2))


if __name__ == "__main__":
    main()
