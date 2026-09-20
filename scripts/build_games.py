#!/usr/bin/env python3
"""Game states as typed decisions, from NanoJev's published corpus.

`C-Tianyu/NanoJev-Data` (CC0-1.0, self-authored programmatic) is the one
public corpus of decisions over game state: mazes, grid navigation,
tic-tac-toe, bag draws, minesweeper. decider and NanoJev both show the same
thing with it — a model of this class can read a board written as text and
put its probability mass on the legal move.

Two kinds of row come out:

* **Deterministic truth.** "Is the square to the north clear?" has one right
  answer that follows from the board. Trained as a normal label.
* **An exact probability law.** A bag draw has a true distribution that can
  be computed, and the corpus carries it in `gold_probs`. Those rows train
  the probability rather than the argmax, which is the part a calibrated
  decision model is actually sold on, and no other source in our mix has a
  gold distribution at all.

Rows whose gold is a teacher's opinion rather than a computed fact are
dropped: `gold_label_kind` must be a deterministic truth or an exact
conditional distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl

REPO = "C-Tianyu/NanoJev-Data"
FILES = [
    "games_v4/data/local_maze_v1/{split}.jsonl",
    "games_v4/data/scaled_games_labeled_v4/{split}.jsonl",
    "games_v4/data/scaled_games_v4b/events/{split}.jsonl",
    "games_v4/data/scaled_games_v4b/policy/{split}.jsonl",
]
EXACT = {"deterministic_truth", "exact_conditional_distribution",
         "exact_distribution", "exact_probability"}
ASKS = ["Is that true of the state?", "Does that hold on this board?",
        "True or false for the position below?"]


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def phrase(name: str) -> str:
    """`clear_north` becomes a claim a reader understands."""
    words = name.replace("_", " ").strip()
    if words.startswith("clear "):
        return f"The square to the {words.split()[-1]} is clear"
    if words.startswith("can "):
        return f"The player {words}"
    if words.startswith("is "):
        return f"It {words}"
    return f"{words[0].upper()}{words[1:]}"


def state_text(row: dict) -> str:
    state = row.get("state")
    if isinstance(state, str) and state.strip():
        return state
    environment = (row.get("metadata") or {}).get("environment_state")
    return json.dumps(environment, indent=1) if environment else ""


def convert(row: dict, rng: random.Random) -> list[Decision]:
    """One decision per question on the board."""
    state = state_text(row)
    if not state:
        return []
    gold = row.get("gold") or {}
    kinds = row.get("gold_label_kind") or {}
    questions = row.get("questions") or {}
    out = []
    for name, answer in gold.items():
        if kinds.get(name) not in EXACT:
            continue
        asked = questions.get(name)
        instruction = (asked.get("instructions")
                       if isinstance(asked, dict) else None)
        if isinstance(answer, bool):
            question = instruction or f"{phrase(name)}. {rng.choice(ASKS)}"
            out.append(Decision(
                id=rid("game", row.get("id", ""), name), state=state,
                question=question,
                options=[Option("Yes", "Yes"), Option("No", "No")],
                label=0 if answer else 1,
                source=f"game-{row.get('family_id', 'state')}", task="noul"))
        elif isinstance(answer, str):
            choices = None
            if isinstance(asked, dict):
                choices = asked.get("options") or asked.get("criteria")
            if isinstance(choices, dict):
                choices = list(choices)
            if not choices or answer not in choices:
                continue
            order = list(choices)
            rng.shuffle(order)
            out.append(Decision(
                id=rid("game", row.get("id", ""), name), state=state,
                question=instruction or "Which move should the player make?",
                options=[Option(str(c), str(c)) for c in order],
                label=order.index(answer),
                source=f"game-{row.get('family_id', 'move')}", task="choice"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/games.jsonl")
    parser.add_argument("--out-test", default="data/test_games.jsonl")
    parser.add_argument("--seed", type=int, default=613)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    rng = random.Random(args.seed)
    train, test = [], []
    for template in FILES:
        for split in ("train", "dev", "calibration", "test", "ood"):
            name = template.format(split=split)
            try:
                path = hf_hub_download(REPO, name, repo_type="dataset",
                                       local_dir="_scratch/nanojev")
            except Exception as error:
                print(f"  {name}: missing ({str(error).splitlines()[0][:60]})")
                continue
            rows = []
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    rows += convert(json.loads(line), rng)
            (test if split in ("test", "ood") else train).extend(rows)
            print(f"  {name}: {len(rows)} decisions", flush=True)

    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    print(f"\nwrote {len(train)} training and {len(test)} test decisions")


if __name__ == "__main__":
    main()
