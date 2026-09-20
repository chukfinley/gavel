#!/usr/bin/env python3
"""Score the independent benchmark that the other projects report on.

`jabr/classifier-benchmark` is the only place where Jev, Von, GLiNER2 and Laya
are measured on identical rows. It now carries two suites, and the other
projects report on both:

    suite   n     Jev            GLiNER2        Von            Laya
    v1      78    0.974 / 0.972  0.795 / 0.785  0.923 / 0.930  0.615 / 0.619
    v2      869   0.964 / 0.966  0.688 / 0.684  0.666 / 0.667  0.585 / 0.583
    both    947   0.965 / 0.967  0.697 / 0.698  0.687 / 0.704  0.587 / 0.588

v2 is the interesting one: it is 49 tasks in domains none of these models was
built for, and it is where Von drops 25.7 points. A broad training mix should
lose less there than a narrow one, so v2 is the number worth optimising.

Reading the question shapes correctly matters more than it sounds. A `Score`
question carries its levels as a **list** of descriptions, one per level, and
the number of levels is the length of that list. Reading it as a dictionary
and falling back to three bare digits, as this script did before, offered only
three options on a five-level task and hid every description from the model.
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

# micro / macro, from results/v1v2-summary.md in the benchmark repository.
PUBLISHED = {
    "v1": {"jev": (0.974, 0.972), "von-1.0.1": (0.923, 0.930),
           "gliner2": (0.795, 0.785), "laya": (0.615, 0.619)},
    "v2": {"jev": (0.964, 0.966), "von-1.0.1": (0.666, 0.667),
           "gliner2": (0.688, 0.684), "laya": (0.585, 0.583)},
    "combined": {"jev": (0.965, 0.967), "von-1.0.1": (0.687, 0.704),
                 "gliner2": (0.697, 0.698), "laya": (0.587, 0.588)},
}
STUB = '''from dataclasses import dataclass, field
@dataclass
class Question:
    instructions: str = ""
@dataclass
class Choice(Question):
    criteria: dict = field(default_factory=dict)
@dataclass
class Noul(Question):
    criteria: dict = field(default_factory=dict)
@dataclass
class Score(Question):
    criteria: list = field(default_factory=list)
    levels: list = field(default_factory=list)
'''


def load_suites(folder: Path) -> dict:
    """The suite imports `von.types`; a stub of the same shape is enough."""
    package = folder / "von"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "types.py").write_text(STUB)
    sys.path.insert(0, str(folder))
    try:
        from bench.suites import SUITES

        return {name: list(tasks) for name, tasks in SUITES.items()}
    except Exception as error:
        print(f"suite registry unavailable ({error}); scanning bench.cases")
    from bench import cases

    tasks = []
    for name in dir(cases):
        member = getattr(cases, name)
        if inspect.isfunction(member) and not name.startswith("_"):
            try:
                task = member()
                if hasattr(task, "cases"):
                    tasks.append(task)
            except Exception:
                continue
    return {"v1": tasks}


def options_for(task):
    """The keys the gold labels use, and the option text the model reads."""
    question = task.question
    criteria = getattr(question, "criteria", None)
    if task.type == "choice":
        keys = list(criteria)
        return keys, [f"{key}: {criteria[key]}" for key in keys]
    if task.type == "noul":
        return [True, False], ["Yes", "No"]
    # A score. The levels are a list of descriptions, one per level, and the
    # gold label is the index into it. A dictionary is accepted as well, so an
    # older or a hand-written suite still works.
    if isinstance(criteria, dict) and criteria:
        keys = list(criteria)
        return keys, [f"{key}: {criteria[key]}" for key in keys]
    levels = list(criteria or []) or list(getattr(question, "levels", None) or [])
    if not levels:
        levels = ["low", "medium", "high"]
    return list(range(len(levels))), [f"{index}: {text}"
                                      for index, text in enumerate(levels)]


def question_text(task) -> str:
    """A noul states a claim; the model is asked whether it holds."""
    instructions = task.question.instructions
    if task.type == "noul":
        return f"{instructions}. Is that true of the state?"
    return instructions


def score_suite(judge, tasks, latencies) -> tuple[list, int, int]:
    per_task, correct, total = [], 0, 0
    for task in tasks:
        keys, options = options_for(task)
        question = question_text(task)
        hits = 0
        for case in task.cases:
            started = time.perf_counter()
            verdict = judge.decide(case.state, question, options)
            latencies.append((time.perf_counter() - started) * 1000)
            chosen = keys[options.index(verdict.option)]
            hits += int(str(chosen).split(":")[0].strip() == str(case.expected))
        per_task.append({"task": task.id, "type": task.type, "correct": hits,
                         "cases": len(task.cases),
                         "accuracy": round(hits / len(task.cases), 4)})
        correct += hits
        total += len(task.cases)
        print(f"  {task.id:26s} {task.type:7s} {hits:3d}/{len(task.cases):<3d} "
              f"= {hits / len(task.cases):.3f}", flush=True)
    return per_task, correct, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chukfinley/gavel-vela-32k")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--suite", default="/tmp/classifier-benchmark")
    parser.add_argument("--suites", default="all",
                        help="v1, v2, or all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    folder = Path(args.suite)
    if not folder.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/jabr/classifier-benchmark", str(folder)],
                       check=True)
    suites = load_suites(folder)
    wanted = list(suites) if args.suites == "all" else \
        [s.strip() for s in args.suites.split(",") if s.strip() in suites]

    from gavel import Gavel

    judge = (Gavel.from_checkpoint(args.checkpoint, args.device, args.max_length)
             if args.checkpoint else
             Gavel.from_pretrained(args.model, args.device, args.max_length))

    latencies: list[float] = []
    report, every_task, all_correct, all_total = {}, [], 0, 0
    for name in wanted:
        print(f"\n=== {name} ({len(suites[name])} tasks, "
              f"{sum(len(t.cases) for t in suites[name])} cases) ===")
        per_task, correct, total = score_suite(judge, suites[name], latencies)
        micro = correct / total
        macro = statistics.mean(t["accuracy"] for t in per_task)
        report[name] = {"micro": micro, "macro": macro, "cases": total,
                        "tasks": per_task, "published": PUBLISHED.get(name, {})}
        every_task += per_task
        all_correct += correct
        all_total += total
        print(f"{name}: micro {micro:.3f}  macro {macro:.3f}  over {total} cases")
        for other, (a, b) in PUBLISHED.get(name, {}).items():
            print(f"    published {other:10s} {a:.3f} / {b:.3f}")

    if len(wanted) > 1:
        micro = all_correct / all_total
        macro = statistics.mean(t["accuracy"] for t in every_task)
        report["combined"] = {"micro": micro, "macro": macro, "cases": all_total,
                              "published": PUBLISHED["combined"]}
        print(f"\ncombined: micro {micro:.3f}  macro {macro:.3f}  over {all_total} cases")
        for other, (a, b) in PUBLISHED["combined"].items():
            print(f"    published {other:10s} {a:.3f} / {b:.3f}")

    report["latency_ms"] = statistics.mean(latencies)
    print(f"latency {statistics.mean(latencies):.0f} ms per case on {judge.device}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
