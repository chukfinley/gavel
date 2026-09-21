#!/usr/bin/env python3
"""How long one decision takes, by option count and state length.

No opinion, just the clock. Runs on whatever device is there and writes
`results/latency_<device>.json`. Two models are timed: the pair scorer
(one sequence per option) and the span head (one sequence per decision),
and for the span head also several questions about one state in one pass,
which is the case the architecture was changed for.

    .venv/bin/python scripts/bench_latency.py [pair model] [span checkpoint]
"""
import json
import statistics
import sys
import time

sys.path.insert(0, "src")
import torch

from gavel import Gavel
from gavel.spanapi import SpanGavel

MODEL = sys.argv[1] if len(sys.argv) > 1 else "chukfinley/gavel-vela-32k"
SPAN = sys.argv[2] if len(sys.argv) > 2 else "export/restore-base/span-head.pt"

WORD = "The quarterly report notes a delay in the payment reconciliation step. "
QUESTION = "Which team should handle this?"
OPTS = ["billing: Refunds, payments, invoices, subscriptions, or unexpected charges",
        "tech: Bugs, errors, crashes, downtime, or integration failures",
        "sales: Pricing questions, quotes, upgrades, demos, or new purchases",
        "account: Login issues, password resets, profile changes, or deletions",
        "other: Anything that does not match the other categories",
        "legal: Contracts, compliance, data processing agreements",
        "ops: Deployment, capacity, on-call, incident response",
        "hr: Hiring, onboarding, payroll, leave"]
EXTRA = [("Is this urgent?", ["Yes", "No"]),
         ("What is the tone?", ["Neutral", "Angry", "Friendly"]),
         ("Does it need a reply today?", ["Yes", "No"])]


def clock(call, repeats: int = 7) -> float:
    call()                                                   # warm up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(times), 1)


def state_of(tokens: int) -> str:
    return (WORD * (tokens // 12 + 1))[:tokens * 5]


def main() -> None:
    pair = Gavel.from_pretrained(MODEL, None, 32768)
    device = str(pair.device).split(":")[0]
    print("device:", pair.device, "threads:", torch.get_num_threads(), flush=True)
    print(f"parameters: {sum(p.numel() for p in pair.model.parameters())/1e6:.0f} M", flush=True)
    span = None
    try:
        span = SpanGavel.from_checkpoint(SPAN, max_length=8192)
        print("span head:", SPAN, flush=True)
    except Exception as error:
        print("no span head:", str(error)[:80], flush=True)

    rows = []
    for state_tokens in (64, 256, 1024, 4096):
        state = state_of(state_tokens)
        for n in (2, 5, 8):
            options = OPTS[:n]
            asked = [(QUESTION, options), *EXTRA]
            row = {"state_tokens": state_tokens, "options": n,
                   "pair_ms": clock(lambda s=state, o=options: pair.decide(s, QUESTION, o))}
            if span is not None:
                row["span_ms"] = clock(lambda s=state, o=options: span.decide(s, QUESTION, o))
                row["span_4q_ms"] = clock(lambda s=state, a=asked: span.decide_many(s, a))
            rows.append(row)
            print("  " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    out = f"results/latency_{device}.json"
    json.dump(rows, open(out, "w"), indent=2)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
