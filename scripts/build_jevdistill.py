#!/usr/bin/env python3
"""MagaBitmex/jev-4b-distill-data as a training source and two test sets.

1500 synthetic support tickets with five typed questions each (route,
intent, sentiment, urgency as a score, escalate as a noul), programmatic
gold, and Jev 1.13's probability per option. Rows keep Jev's distribution
in `meta.teacher`, so the trainers' `--teacher-file` machinery applies
without a request. The eval and hard splits become held-out test sets.

    .venv/bin/python scripts/build_jevdistill.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("_scratch/jev4b")
YES = {"yes", "true"}


def fetch() -> None:
    """The dataset is on the Hub; a pod does not have the local snapshot."""
    if (ROOT / "data3" / "train.jsonl").exists():
        return
    import os

    from huggingface_hub import hf_hub_download, snapshot_download

    token = os.environ.get("HF_TOKEN")
    try:
        snapshot_download("MagaBitmex/jev-4b-distill-data", repo_type="dataset", local_dir=str(ROOT), token=token)
    except Exception as error:
        # The upstream repo answered 404 on 2026-09-22 (gone or private). Our
        # own copy of the six files sits in the results dataset, Apache 2.0.
        print("upstream dataset unavailable, using our copy:", str(error)[:80])
        repo = os.environ.get("RESULTS_REPO", "chukfinley/gavel-runs")
        for name in ("data3/train.jsonl", "data3/train_teacher.jsonl", "data3/eval.jsonl",
                     "data3/eval_teacher.jsonl", "data_hard/eval.jsonl", "data_hard/eval_teacher.jsonl"):
            local = hf_hub_download(repo, f"data/jev4b/{name}", repo_type="dataset",
                                    local_dir=str(ROOT / ".mirror"), token=token)
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(open(local, "rb").read())


def rows_of(file: Path, teacher_file: Path | None, source: str) -> list[dict]:
    teachers = {}
    if teacher_file and teacher_file.exists():
        for line in open(teacher_file):
            record = json.loads(line)
            teachers[record["state_id"]] = record.get("teacher") or {}
    out = []
    for line in open(file):
        record = json.loads(line)
        gold = record.get("gold") or {}
        teacher = teachers.get(record["state_id"], {})
        for name, question in (record.get("questions") or {}).items():
            answer = gold.get(name)
            if answer is None:
                continue
            kind = question["type"]
            if kind == "choice":
                criteria = question["criteria"]
                keys = list(criteria)
                options = [{"id": k, "description": f"{k}: {criteria[k]}"} for k in keys]
                if str(answer) not in keys:
                    continue
                label = keys.index(str(answer))
                probabilities = (teacher.get(name) or {}).get("probabilities")
                soft = [float(probabilities.get(k, 0.0)) for k in keys] if probabilities else None
                task = "choice"
            elif kind == "score":
                levels = question["criteria"]
                keys = [str(i) for i in range(len(levels))]
                options = [{"id": k, "description": f"{k}: {text}"} for k, text in zip(keys, levels)]
                # Gold is a word ("medium"); the teacher's legend maps index to text.
                legend = (teacher.get(name) or {}).get("legend") or {}
                index = None
                if isinstance(answer, int | float):
                    index = round(float(answer))
                else:
                    words = {"low": 0, "medium": 1, "high": 2}
                    if len(levels) == 3 and str(answer).lower() in words:
                        index = words[str(answer).lower()]
                    for k, text in legend.items():
                        if str(answer).lower() in str(text).lower():
                            index = int(k)
                if index is None or not 0 <= index < len(levels):
                    continue
                label = index
                probabilities = (teacher.get(name) or {}).get("probabilities")
                soft = [float(probabilities.get(k, 0.0)) for k in keys] if probabilities else None
                task = "score"
            else:  # noul
                criteria = question.get("criteria") or {}
                options = [{"id": "Yes", "description": criteria.get("true", "Yes")},
                           {"id": "No", "description": criteria.get("false", "No")}]
                truth = str(answer).lower() in YES if not isinstance(answer, bool) else answer
                label = 0 if truth else 1
                p = (teacher.get(name) or {}).get("noul")
                soft = [float(p), 1.0 - float(p)] if p is not None else None
                task = "noul"
            if soft and abs(sum(soft) - 1.0) > 0.05:
                total = sum(soft) or 1.0
                soft = [v / total for v in soft]
            row_id = hashlib.sha1(f"jev4b|{record['state_id']}|{name}".encode()).hexdigest()[:20]
            out.append({"id": row_id, "state": record["state"], "question": question["instructions"],
                        "options": options, "label": label, "source": source, "task": task,
                        "meta": {"teacher": soft} if soft else {}})
    return out


def write(path: str, rows: list[dict]) -> None:
    with open(path, "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    print(f"{path}: {len(rows)} rows, {sum(1 for r in rows if r['meta'].get('teacher'))} with teacher")


def main() -> None:
    fetch()
    write("data/jevdistill.jsonl", rows_of(ROOT / "data3/train.jsonl", ROOT / "data3/train_teacher.jsonl", "jevdistill"))
    write("data/test_jevdistill.jsonl", rows_of(ROOT / "data3/eval.jsonl", ROOT / "data3/eval_teacher.jsonl", "jevdistill-eval"))
    write("data/test_jevdistill_hard.jsonl", rows_of(ROOT / "data_hard/eval.jsonl", ROOT / "data_hard/eval_teacher.jsonl", "jevdistill-hard"))


if __name__ == "__main__":
    main()
