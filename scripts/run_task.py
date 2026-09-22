#!/usr/bin/env python3
"""Answer a fixed-question task item by item and write a trace as it goes.

    .venv/bin/python scripts/run_task.py --task emails --n 20 --jev --trace-dir _scratch/traces/x
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from gavel.tasks import gold_of, load_items, questions_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("emails", "products"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--span", default=None)
    parser.add_argument("--jev", action="store_true")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--record", default="", help="append the answers as training rows")
    args = parser.parse_args()
    folder = Path(args.trace_dir)
    folder.mkdir(parents=True, exist_ok=True)
    trace = open(folder / "trace.jsonl", "a")

    def emit(**record) -> None:
        trace.write(json.dumps(record) + "\n")
        trace.flush()

    who = "typesafe/jev-1.13" if args.jev else (args.span or args.model)
    emit(kind="start", task=args.task, n=args.n, model=who)
    if args.jev:
        from gavel.jevapi import JevGavel
        judge = JevGavel()
    elif args.span:
        from gavel.spanapi import SpanGavel
        judge = SpanGavel.from_checkpoint(args.span, max_length=4096)
    else:
        from gavel import Gavel
        judge = Gavel.from_pretrained(args.model, None, 2048)
    emit(kind="loaded", device=str(judge.device))
    recorder = open(args.record, "a") if args.record else None

    items = load_items(args.task, args.n, args.offset)
    for index, item in enumerate(items):
        questions = questions_for(args.task, item)
        started = time.perf_counter()
        try:
            if args.jev:
                verdicts = judge.decide_many(item["state"], [(q, o) for _, q, o, _ in questions],
                                             noul=all(k == "noul" for *_, k in questions))
            else:
                verdicts = [judge.decide(item["state"], q, o) for _, q, o, _ in questions]
        except Exception as error:
            emit(kind="error", index=index, message=str(error)[:200])
            continue
        ms = round((time.perf_counter() - started) * 1000, 1)
        answers = {}
        for (name, question, options, kind), verdict in zip(questions, verdicts):
            gold = gold_of(args.task, item, name, options)
            answers[name] = {"option": verdict.option, "confidence": verdict.confidence,
                             "probabilities": verdict.probabilities,
                             "gold": options[gold] if gold is not None else None}
            if recorder is not None:
                recorder.write(json.dumps({
                    "id": hashlib.sha1(f"task|{args.task}|{item['id']}|{name}".encode()).hexdigest()[:20],
                    "state": item["state"], "question": question,
                    "options": [{"id": f"o{k}", "description": o} for k, o in enumerate(options)],
                    "label": options.index(verdict.option), "gold": gold,
                    "source": f"task-{args.task}-{'jev' if args.jev else 'ours'}", "task": kind,
                    "meta": {"teacher": [verdict.probabilities.get(o, 0.0) for o in options]
                             if args.jev else None, "item": item["id"]}}) + "\n")
                recorder.flush()
        emit(kind="item", index=index, item=item["id"], state=item["state"][:400], answers=answers, ms=ms,
             usd=round(getattr(judge, "spent_usd", 0.0), 5))
    emit(kind="done", items=len(items))


if __name__ == "__main__":
    main()
