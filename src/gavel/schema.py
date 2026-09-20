"""The one record format that every source is converted into.

The format follows the OpenJev fixture, thus their committed evaluation rows
can be read without a converter:

    {"id": str, "state": str, "question": str,
     "options": [{"id": str, "description": str}], "label": int}

`label` is the index into `options`. `state` holds the unstructured situation,
`question` the criterion, and `options` the typed answer space. The option
texts arrive with the record, thus the answer space is defined at run time and
not baked into the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass(slots=True)
class Option:
    id: str
    description: str


@dataclass(slots=True)
class Decision:
    id: str
    state: str
    question: str
    options: list[Option]
    label: int | None = None
    source: str = ""
    task: str = "choice"          # choice | noul | score
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        row = {
            "id": self.id,
            "state": self.state,
            "question": self.question,
            "options": [{"id": o.id, "description": o.description} for o in self.options],
            "source": self.source,
            "task": self.task,
        }
        if self.label is not None:
            row["label"] = self.label
        if self.meta:
            row["meta"] = self.meta
        return row


def from_json(row: dict) -> Decision:
    return Decision(
        id=str(row["id"]),
        state=row["state"] if isinstance(row["state"], str) else json.dumps(row["state"]),
        question=row["question"],
        options=[Option(str(o["id"]), o["description"]) for o in row["options"]],
        label=row.get("label"),
        source=row.get("source", ""),
        task=row.get("task", "choice"),
        meta=row.get("meta", {}),
    )


def read_jsonl(path: str | Path) -> Iterator[Decision]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield from_json(json.loads(line))


def write_jsonl(path: str | Path, rows: list[Decision]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")
