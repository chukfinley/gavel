#!/usr/bin/env python3
"""How long one decision takes, by option count and state length.

No opinion, just the clock. Runs on whatever device is there.
"""
import sys, time, statistics, json
sys.path.insert(0, "src")
import torch
from gavel import Gavel

MODEL = "chukfinley/gavel-vela-32k"
judge = Gavel.from_pretrained(MODEL, None, 32768)
print("device:", judge.device, "threads:", torch.get_num_threads(), flush=True)
total = sum(p.numel() for p in judge.model.parameters())
print(f"parameters: {total/1e6:.0f} M", flush=True)

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

rows = []
for state_tokens in (64, 256, 1024, 4096):
    state = (WORD * (state_tokens // 12 + 1))[:state_tokens * 5]
    for n_options in (2, 5, 8):
        judge.decide(state, QUESTION, OPTS[:n_options])          # warm up
        times = []
        for _ in range(5):
            start = time.perf_counter()
            judge.decide(state, QUESTION, OPTS[:n_options])
            times.append((time.perf_counter() - start) * 1000)
        ms = statistics.median(times)
        rows.append({"state_tokens": state_tokens, "options": n_options, "ms": round(ms, 1)})
        print(f"  state~{state_tokens:>5} tok x {n_options} options: {ms:7.1f} ms", flush=True)

json.dump(rows, open("results/latency_cpu.json", "w"), indent=2)
print("done", flush=True)
