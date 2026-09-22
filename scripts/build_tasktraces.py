#!/usr/bin/env python3
"""The closed model's answers on the task items, as a training source.

Reads `data/traces/tasks.jsonl`, keeps the rows Jev answered, and writes
`data/tasktraces.jsonl` with Jev's distribution in `meta.teacher`. Where
the dataset had gold (spam, product category) the gold wins the label.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    source = Path("data/traces/tasks.jsonl")
    rows: dict[str, dict] = {}
    if source.exists():
        for line in open(source):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not row.get("source", "").endswith("-jev"):
                continue
            label = row["gold"] if row.get("gold") is not None else row["label"]
            rows[row["id"]] = {"id": row["id"], "state": row["state"], "question": row["question"],
                               "options": row["options"], "label": label,
                               "source": row["source"].replace("-jev", ""), "task": row["task"],
                               "meta": {"teacher": row["meta"].get("teacher")} if row.get("meta", {}).get("teacher") else {}}
    with open("data/tasktraces.jsonl", "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows.values())
    print(f"wrote {len(rows)} task rows from the closed model")


if __name__ == "__main__":
    main()
