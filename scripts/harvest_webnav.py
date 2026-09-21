#!/usr/bin/env python3
"""Let the closed model solve navigation tasks; keep every decision as a row.

The rows land in `data/traces/webnav.jsonl` (source `webnav-jev`, label =
Jev's choice, `meta.teacher` = its probabilities) and `build_webnav.py`
folds them into the mix. Costs cents: a page is one request of a few
thousand input tokens.

    .venv/bin/python scripts/harvest_webnav.py --tasks scripts/webnav_tasks.jsonl --rounds 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="scripts/webnav_tasks.jsonl")
    parser.add_argument("--rounds", type=int, default=1, help="repeat the list this many times")
    parser.add_argument("--hops", type=int, default=6)
    parser.add_argument("--record", default="data/traces/webnav.jsonl")
    parser.add_argument("--ours", action="store_true", help="run our model instead of Jev")
    args = parser.parse_args()
    tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
    solved = total = 0
    for round_index in range(args.rounds):
        for task in tasks:
            stamp = time.strftime("%m%d-%H%M%S")
            folder = Path("_scratch/traces") / f"harvest-{stamp}"
            command = [sys.executable, "scripts/live_webnav.py", "--url", task["url"], "--goal", task["goal"],
                       "--hops", str(args.hops), "--trace-dir", str(folder), "--record", args.record]
            if task.get("pattern"):
                command += ["--pattern", task["pattern"]]
            if not args.ours:
                command.append("--jev")
            subprocess.run(command, check=False, timeout=900,
                           stdout=open(folder.parent / f"harvest-{stamp}.log", "w"), stderr=subprocess.STDOUT)
            found = False
            trace = folder / "trace.jsonl"
            if trace.exists():
                for line in open(trace):
                    record = json.loads(line)
                    if record.get("kind") == "done":
                        found = bool(record.get("found"))
            total += 1
            solved += found
            print(f"[{stamp}] {'ok ' if found else 'no '} {task['url']} -> {task['goal']}", flush=True)
    print(f"solved {solved}/{total}", flush=True)


if __name__ == "__main__":
    main()
