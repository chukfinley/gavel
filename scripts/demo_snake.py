#!/usr/bin/env python3
"""Snake, played with the decision model as the "smart if statement".

The game state is text, exactly as the training rows word it. At every
step the model is asked, for each legal direction, whether that move
avoids a collision. The safe moves are then ordered by distance to the
food and the first is taken. The model never generates a token: it
answers three yes/no questions from one reading of the board.

Three players run on the same boards so the numbers mean something:

* `random`: a legal move at random. The floor.
* `model`: safety from the model, then the food heuristic.
* `oracle`: safety from the simulator itself. The ceiling for this policy.

    .venv/bin/python scripts/demo_snake.py [--span runs/x/best.pt] [--games 20]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time

sys.path.insert(0, "src")

DIRECTIONS = {"north": (-1, 0), "south": (1, 0), "east": (0, 1), "west": (0, -1)}
REVERSE = {"north": "south", "south": "north", "east": "west", "west": "east"}


class Snake:
    def __init__(self, size: int, rng: random.Random):
        self.size = size
        self.rng = rng
        middle = size // 2
        self.body = [(middle, middle), (middle, middle - 1), (middle, middle - 2)]
        self.direction = "east"
        self.food = self.place_food()
        self.eaten = 0

    def place_food(self):
        free = [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in self.body]
        return self.rng.choice(free) if free else None

    def legal(self) -> list[str]:
        return [d for d in DIRECTIONS if d != REVERSE[self.direction]]

    def safe(self, direction: str) -> bool:
        """The ground truth the model is asked to reproduce."""
        dr, dc = DIRECTIONS[direction]
        r, c = self.body[0][0] + dr, self.body[0][1] + dc
        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        grows = (r, c) == self.food
        occupied = self.body if grows else self.body[:-1]
        return (r, c) not in occupied

    def step(self, direction: str) -> bool:
        if not self.safe(direction):
            return False
        dr, dc = DIRECTIONS[direction]
        head = (self.body[0][0] + dr, self.body[0][1] + dc)
        self.body.insert(0, head)
        if head == self.food:
            self.eaten += 1
            self.food = self.place_food()
        else:
            self.body.pop()
        self.direction = direction
        return True

    def text(self) -> str:
        body = json.dumps([list(p) for p in self.body])
        food = json.dumps(list(self.food)) if self.food else "none"
        return (f"Snake on a {self.size}x{self.size} board. Coordinates are zero-based "
                f"(row,column); north decreases row and east increases column. "
                f"Body in head-first order: {body}. Direction: {self.direction}. "
                f"Food: {food}. Reverse moves are disallowed. A move onto food grows "
                f"the body; otherwise the tail vacates. Entering that vacated tail cell "
                f"is allowed. Walls and occupied body cells cause collision.")

    def distance_after(self, direction: str) -> int:
        dr, dc = DIRECTIONS[direction]
        r, c = self.body[0][0] + dr, self.body[0][1] + dc
        return abs(r - self.food[0]) + abs(c - self.food[1]) if self.food else 0


def question(direction: str) -> str:
    return (f"If the snake takes {direction}, will it avoid wall and body collision on "
            f"this next step? Eating food retains the tail; otherwise the tail vacates "
            f"and that vacated cell is safe to enter. Completing the board counts as safe.")


def play(game: Snake, judge, kind: str, max_steps: int, clock: list[float],
         confusion: dict[str, int]) -> dict:
    steps = 0
    while steps < max_steps and game.food is not None:
        legal = game.legal()
        truth = {d: game.safe(d) for d in legal}
        if kind == "random":
            verdicts = {d: True for d in legal}
        elif kind == "oracle":
            verdicts = truth
        else:
            state = game.text()
            start = time.perf_counter()
            if hasattr(judge, "decide_many"):
                answers = judge.decide_many(state, [(question(d), ["Yes", "No"]) for d in legal])
            else:
                answers = [judge.decide(state, question(d), ["Yes", "No"]) for d in legal]
            clock.append((time.perf_counter() - start) * 1000)
            verdicts = {d: a.option == "Yes" for d, a in zip(legal, answers)}
            for d in legal:
                confusion["right" if verdicts[d] == truth[d] else "wrong"] += 1
        safe = [d for d in legal if verdicts[d]] or legal
        if kind == "random":
            move = game.rng.choice(safe)
        else:
            move = min(safe, key=game.distance_after)
        if not game.step(move):
            break
        steps += 1
    return {"steps": steps, "eaten": game.eaten}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--span", default=None, help="span checkpoint; default: pair model")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--out", default="results/demo_snake.json")
    args = parser.parse_args()

    if args.span:
        from gavel.spanapi import SpanGavel
        judge = SpanGavel.from_checkpoint(args.span, max_length=2048)
    else:
        from gavel import Gavel
        judge = Gavel.from_pretrained(args.model, None, 2048)

    report = {}
    for kind in ("random", "model", "oracle"):
        clock: list[float] = []
        confusion = {"right": 0, "wrong": 0}
        results = [play(Snake(args.size, random.Random(seed)), judge, kind,
                        args.max_steps, clock, confusion)
                   for seed in range(args.games)]
        row = {"steps": statistics.mean(r["steps"] for r in results),
               "eaten": statistics.mean(r["eaten"] for r in results)}
        if clock:
            row["ms_per_turn"] = round(statistics.median(clock), 1)
            total = confusion["right"] + confusion["wrong"]
            row["safety_accuracy"] = round(confusion["right"] / max(total, 1), 3)
            row["turns"] = total // 3
        report[kind] = row
        print(f"{kind:>7}: " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    json.dump({"model": args.span or args.model, "size": args.size, **report},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
