#!/usr/bin/env python3
"""Tool selection: which function should handle this request.

This is the closed half of an agent loop. The tool list arrives with the
request, thus the answer space changes on every call, which is exactly the
shape this model is built for. The open half — what arguments to pass — stays
with a language model, because a decision model cannot write free text.

An extra option, "answer without a tool", is always present. Agents that call
a tool for every message waste time and money, and the model should be able to
say that none of the offered tools fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl  # noqa: E402

NO_TOOL = Option("no_tool", "No tool is needed, answer directly")


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def describe(tool: dict) -> tuple[str, str]:
    function = tool.get("function", tool)
    name = str(function.get("name", "")).strip()
    description = str(function.get("description", "")).strip().split("\n")[0]
    return name, (f"{name}: {description}" if description else name)


HERMES_CONFIGS = ["func_calling_singleturn", "func_calling", "glaive_func_calling",
                  "json_mode_singleturn", "json_mode_agentic"]


def hermes_rows(limit: int, rng: random.Random) -> list[Decision]:
    """All subsets of the collection, not only the default one."""
    from datasets import load_dataset

    parts = []
    for config in HERMES_CONFIGS:
        try:
            parts.append(load_dataset("NousResearch/hermes-function-calling-v1",
                                      config, split="train"))
        except Exception:                                        # noqa: BLE001
            continue
    if not parts:
        parts = [load_dataset("NousResearch/hermes-function-calling-v1", split="train")]
    data = [row for part in parts for row in part]
    out = []
    for index, row in enumerate(data):
        if len(out) >= limit:
            break
        try:
            tools = row["tools"]
            tools = json.loads(tools) if isinstance(tools, str) else tools
        except Exception:                                        # noqa: BLE001
            continue
        if not isinstance(tools, list) or len(tools) < 2:
            continue
        conversation = row["conversations"]
        user_turn = next((turn["value"] for turn in conversation
                          if turn.get("from") in ("human", "user")), None)
        called = None
        for turn in conversation:
            value = str(turn.get("value", ""))
            found = re.search(r'"name"\s*:\s*"([^"]+)"', value)
            if turn.get("from") in ("gpt", "assistant") and found:
                called = found.group(1)
                break
        if not user_turn or not called:
            continue
        options, names = [], []
        for tool in tools:
            name, text = describe(tool)
            if name:
                options.append(Option(name, text))
                names.append(name)
        if called not in names or len(options) < 2:
            continue
        options.append(NO_TOOL)
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        out.append(Decision(
            id=rid("hermes-tools", str(index)),
            state=str(user_turn).strip()[:2000],
            question="Which tool should handle this request?",
            options=shuffled,
            label=[o.id for o in shuffled].index(called),
            source="tools-hermes", task="choice"))
    return out


def json_objects(text: str) -> list[dict]:
    """Every top-level JSON object in a text, found by matching braces.

    The tool definitions are pretty-printed and contain nested objects, thus a
    regular expression cannot delimit them.
    """
    found, depth, start = [], 0, None
    for position, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = position
            depth += 1
        elif character == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    found.append(json.loads(text[start : position + 1]))
                except Exception:                                # noqa: BLE001
                    pass
                start = None
    return found


def glaive_rows(limit: int, rng: random.Random) -> list[Decision]:
    from datasets import load_dataset

    data = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
    out, no_tool = [], 0
    for index, row in enumerate(data):
        if len(out) >= limit:
            break
        system = str(row.get("system", ""))
        chat = str(row.get("chat", ""))
        tools = [t for t in json_objects(system) if "name" in t]
        user = re.search(r"USER:\s*(.+?)(?:ASSISTANT:|$)", chat, re.S)
        called = re.search(r"<functioncall>\s*\{\s*\"name\"\s*:\s*\"([^\"]+)\"", chat)
        if not user or not tools:
            continue
        options = []
        for tool in tools:
            name, text = describe(tool)
            if name:
                options.append(Option(name, text))
        if len(options) < 1:
            continue
        options.append(NO_TOOL)
        order = list(range(len(options)))
        rng.shuffle(order)
        shuffled = [options[i] for i in order]
        names = [o.id for o in shuffled]
        if called and called.group(1) in names:
            label = names.index(called.group(1))
        else:
            # The assistant answered without calling anything: that is the
            # "no tool" case, and those rows keep the option honest.
            if no_tool > limit // 3:
                continue
            no_tool += 1
            label = names.index("no_tool")
        out.append(Decision(
            id=rid("glaive-tools", str(index)),
            state=user.group(1).strip()[:2000],
            question="Which tool should handle this request?",
            options=shuffled, label=label, source="tools-glaive", task="choice"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/tools.jsonl")
    parser.add_argument("--out-test", default="data/test_tools.jsonl")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--test-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=131)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[Decision] = []
    for name, builder in [("hermes", hermes_rows), ("glaive", glaive_rows)]:
        try:
            produced = builder(args.limit, rng)
            rows.extend(produced)
            print(f"  {name}: {len(produced)}", flush=True)
        except Exception as error:                               # noqa: BLE001
            print(f"  {name} failed: {error}", flush=True)

    rng.shuffle(rows)
    write_jsonl(args.out_test, rows[: args.test_rows])
    write_jsonl(args.out_train, rows[args.test_rows :])
    print(f"\ntrain {len(rows) - args.test_rows}  test {args.test_rows}")


if __name__ == "__main__":
    main()
