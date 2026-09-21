#!/usr/bin/env python3
"""Snake safety rows from the simulator, balanced, as many as wanted.

The 1356 NanoJev snake rows are 94 percent "Yes", and both our models
learned to answer "Yes" to everything (0.078 accuracy against the
simulator on 2026-09-21). The simulator knows the truth for free, so this
writes boards where half the asked moves collide: walls, the body, and
the tail cell that is safe only because it vacates.

    .venv/bin/python scripts/build_snake.py --rows 20000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from demo_snake import DIRECTIONS, Snake, question


def random_game(rng: random.Random) -> Snake:
    size = rng.choice([8, 12, 16, 32])
    game = Snake(size, rng)
    # Walk a while so the body has a shape; prefer safe moves, stop at a dead end.
    for _ in range(rng.randint(0, 3 * size)):
        safe = [d for d in game.legal() if game.safe(d)]
        if not safe:
            break
        move = rng.choice(safe)
        if game.food and rng.random() < 0.6:
            move = min(safe, key=game.distance_after)
        game.step(move)
    return game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--out", default="data/snake.jsonl")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows, want_yes = [], True
    while len(rows) < args.rows:
        game = random_game(rng)
        options = [(d, game.safe(d)) for d in game.legal()]
        matching = [d for d, safe in options if safe == want_yes]
        if not matching:
            continue
        direction = rng.choice(matching)
        state = game.text()
        rows.append({"id": hashlib.sha1(f"{state}|{direction}".encode()).hexdigest()[:20],
                     "state": state, "question": question(direction),
                     "options": [{"id": "Yes", "description": "Yes"}, {"id": "No", "description": "No"}],
                     "label": 0 if want_yes else 1, "source": "snake-sim", "task": "noul",
                     "meta": {"direction": direction, "size": game.size}})
        want_yes = not want_yes
    unique = {row["id"]: row for row in rows}
    with open(args.out, "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in unique.values())
    yes = sum(row["label"] == 0 for row in unique.values())
    print(f"wrote {len(unique)} snake rows, {yes} yes / {len(unique) - yes} no; "
          f"moves {DIRECTIONS and list(DIRECTIONS)}")


if __name__ == "__main__":
    main()
