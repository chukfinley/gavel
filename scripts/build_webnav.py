#!/usr/bin/env python3
"""The navigation rows the closed model produced, as a training source.

Reads `data/traces/webnav.jsonl`, keeps the rows Jev decided (our own
model's rows are traces, not labels), drops duplicates and states that
carry no links, writes `data/webnav.jsonl` for `assemble.py`.
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    source = Path("data/traces/webnav.jsonl")
    rows, seen = [], set()
    if source.exists():
        for line in open(source):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("source") != "webnav-jev" or row["id"] in seen or len(row.get("options", [])) < 2:
                continue
            seen.add(row["id"])
            teacher = (row.get("meta") or {}).get("teacher")
            rows.append({"id": row["id"], "state": row["state"], "question": row["question"],
                         "options": row["options"], "label": row["label"], "source": "webnav",
                         "task": "choice", "meta": {"teacher": teacher} if teacher else {}})
    with open("data/webnav.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} navigation rows")


if __name__ == "__main__":
    main()
