#!/usr/bin/env python3
"""Score the independent benchmark that the other projects report on.

`jabr/classifier-benchmark` is the only place where Jev, Von, GLiNER2 and Laya
are measured on identical rows, so it is the only comparable number this
project can produce. Published results, v1 (8 tasks, 78 cases):

    Jev 0.974 micro / 0.972 macro     Von 1.0.1 0.923 / 0.930
    GLiNER2 0.795 / 0.785             Laya 0.615 / 0.619
"""

from __future__ import annotations

import argparse
import inspect
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PUBLISHED = {"jev": (0.974, 0.972), "von-1.0.1": (0.923, 0.930),
             "gliner2": (0.795, 0.785), "laya": (0.615, 0.619)}
STUB = '''from dataclasses import dataclass, field
@dataclass
class Question:
    instructions: str = ""
@dataclass
class Choice(Question):
    criteria: dict = field(default_factory=dict)
@dataclass
class Noul(Question):
    pass
@dataclass
class Score(Question):
    levels: list = field(default_factory=list)
    criteria: dict = field(default_factory=dict)
'''


def load_tasks(folder: Path):
    """The suite imports `von.types`; a stub of the same shape is enough."""
    package = folder / "von"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "types.py").write_text(STUB)
    sys.path.insert(0, str(folder))
    from bench import cases

    tasks = []
    for name in dir(cases):
        member = getattr(cases, name)
        if inspect.isfunction(member) and not name.startswith("_"):
            try:
                task = member()
                if hasattr(task, "cases"):
                    tasks.append(task)
            except Exception:                                    # noqa: BLE001
                continue
    return tasks


def options_for(task):
    """Keys the gold labels use, and the option text the model reads."""
    question = task.question
    criteria = getattr(question, "criteria", None)
    if task.type == "choice":
        keys = list(criteria)
        return keys, [f"{k}: {criteria[k]}" for k in keys]
    if task.type == "noul":
        return [True, False], ["Yes", "No"]
    if isinstance(criteria, dict):
        keys = list(criteria)
        return keys, [f"{k}: {criteria[k]}" for k in keys]
    levels = getattr(question, "levels", None) or [0, 1, 2]
    return list(range(len(levels))), [str(level) for level in levels]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--suite", default="/tmp/classifier-benchmark")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    folder = Path(args.suite)
    if not folder.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/jabr/classifier-benchmark", str(folder)],
                       check=True)
    tasks = load_tasks(folder)

    from gavel import Gavel

    judge = (Gavel.from_checkpoint(args.checkpoint, args.device, args.max_length)
             if args.checkpoint else
             Gavel.from_pretrained(args.model, args.device, args.max_length))

    per_task, latencies, correct, total = [], [], 0, 0
    for task in tasks:
        keys, options = options_for(task)
        hits = 0
        for case in task.cases:
            started = time.perf_counter()
            verdict = judge.decide(case.state, task.question.instructions, options)
            latencies.append((time.perf_counter() - started) * 1000)
            chosen = keys[options.index(verdict.option)]
            hits += int(str(chosen).split(":")[0].strip() == str(case.expected))
        per_task.append({"task": task.id, "type": task.type,
                         "correct": hits, "cases": len(task.cases),
                         "accuracy": round(hits / len(task.cases), 4)})
        correct += hits
        total += len(task.cases)
        print(f"  {task.id:22s} {task.type:7s} {hits}/{len(task.cases)} = {hits / len(task.cases):.3f}")

    micro = correct / total
    macro = statistics.mean(t["accuracy"] for t in per_task)
    print(f"\nmicro {micro:.3f}  macro {macro:.3f}  "
          f"latency {statistics.mean(latencies):.0f} ms on {judge.device}")
    for name, (a, b) in PUBLISHED.items():
        print(f"  published {name:10s} {a:.3f} / {b:.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"micro": micro, "macro": macro, "tasks": per_task,
             "latency_ms": statistics.mean(latencies), "published": PUBLISHED}, indent=2))


if __name__ == "__main__":
    main()
