#!/usr/bin/env python3
"""One snake game, every turn written out as the model plays it.

Same rules and wording as `demo_snake.py`; this one writes a JSON line
per turn with the three safety verdicts, their probabilities, the truth
from the simulator and the move taken, so the WebUI can draw the board
while the game runs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from demo_snake import Snake, question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--span", default=None)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--bundled", action="store_true")
    parser.add_argument("--trace-dir", required=True)
    args = parser.parse_args()
    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = open(trace_dir / "trace.jsonl", "a")

    def emit(**record) -> None:
        trace.write(json.dumps(record) + "\n")
        trace.flush()

    emit(kind="start", size=args.size, seed=args.seed, model=args.span or args.model,
         bundled=args.bundled and bool(args.span))
    if args.span:
        from gavel.spanapi import SpanGavel
        judge = SpanGavel.from_checkpoint(args.span, max_length=2048)
    else:
        from gavel import Gavel
        judge = Gavel.from_pretrained(args.model, None, 2048)
    emit(kind="loaded", device=str(judge.device))

    game = Snake(args.size, random.Random(args.seed))
    for step in range(1, args.max_steps + 1):
        if game.food is None:
            break
        legal = game.legal()
        truth = {d: game.safe(d) for d in legal}
        state = game.text()
        started = time.perf_counter()
        if args.bundled and args.span:
            answers = judge.decide_many(state, [(question(d), ["Yes", "No"]) for d in legal])
        else:
            answers = [judge.decide(state, question(d), ["Yes", "No"]) for d in legal]
        ms = round((time.perf_counter() - started) * 1000, 1)
        verdicts = {d: {"safe": a.option == "Yes", "p_yes": a.probabilities.get("Yes", 0.0),
                        "truth": truth[d]} for d, a in zip(legal, answers)}
        safe = [d for d in legal if verdicts[d]["safe"]] or legal
        move = min(safe, key=game.distance_after)
        body_before, food_before = [list(p) for p in game.body], list(game.food)
        alive = game.step(move)
        emit(kind="step", step=step, body=body_before, food=food_before, direction=game.direction,
             verdicts=verdicts, move=move, alive=alive, eaten=game.eaten, ms=ms,
             body_after=[list(p) for p in game.body])
        if not alive:
            break
    emit(kind="done", steps=step, eaten=game.eaten)


if __name__ == "__main__":
    main()
