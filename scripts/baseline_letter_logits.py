#!/usr/bin/env python3
"""The OpenJev baseline, re-implemented, so it can run on our own test set.

OpenJev publishes numbers for its four fixtures only. To compare on the wider
test set, the same method must run here: one forward pass through a causal
language model, then read the logits of the single tokens "A", "B", "C" ... at
the last position and softmax over exactly those.

This follows `openjev/src/openjev_phase1/direct.py`, including the check that
each answer letter is one exact round-trip token.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gavel.losses import balanced_accuracy, expected_calibration_error  # noqa: E402
from gavel.schema import read_jsonl                                    # noqa: E402

LETTERS = [chr(ord("A") + i) for i in range(26)]


def slot_ids(tokenizer, count: int) -> list[int]:
    ids = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"{letter!r} is not one exact round-trip token")
        ids.append(encoded[0])
    return ids


def build_prompt(tokenizer, decision) -> str:
    lines = [f"Situation:\n{decision.state}\n" if decision.state else "",
             f"Question: {decision.question}", "", "Options:"]
    for letter, option in zip(LETTERS, decision.options):
        lines.append(f"{letter}. {option.description}")
    lines.append("")
    lines.append("Answer with the letter of the correct option.")
    messages = [{"role": "user", "content": "\n".join(line for line in lines if line is not None)}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map="cuda")
    model.eval()

    groups: dict[str, list] = defaultdict(list)
    for row in read_jsonl(args.test):
        groups[row.source or "all"].append(row)

    report = {}
    for source, rows in sorted(groups.items()):
        probabilities, labels = [], []
        for row in rows:
            prompt = build_prompt(tokenizer, row)
            ids = tokenizer.encode(prompt, add_special_tokens=False)[-args.max_length:]
            slots = slot_ids(tokenizer, len(row.options))
            tensor = torch.tensor([ids], device="cuda")
            logits = model(input_ids=tensor).logits[0, -1, :].float()
            probabilities.append(torch.softmax(logits[slots], dim=-1).cpu())
            labels.append(row.label)
        width = max(p.size(0) for p in probabilities)
        stacked = torch.stack([torch.nn.functional.pad(p, (0, width - p.size(0))) for p in probabilities])
        target = torch.tensor(labels)
        prediction = stacked.argmax(dim=-1)
        report[source] = {
            "rows": len(rows),
            "accuracy": (prediction == target).float().mean().item(),
            "balanced_accuracy": balanced_accuracy(prediction, target),
            "ece": expected_calibration_error(stacked, target),
        }
        print(f"{source:22s} rows {len(rows):>4d}  acc {report[source]['accuracy']:.4f}  "
              f"bal.acc {report[source]['balanced_accuracy']:.4f}  ece {report[source]['ece']:.4f}",
              flush=True)

    mean = sum(v["balanced_accuracy"] for v in report.values()) / len(report)
    print(f"\nmean over {len(report)} strata: {mean:.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"model": args.model, "strata": report, "mean": mean}, indent=2))


if __name__ == "__main__":
    main()
