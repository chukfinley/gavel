"""Fixed-question tasks over many items: the field's own demos.

An email gets the same four questions every time; a product gets one
question with the shop's categories as options. Both models answer every
item, the WebUI shows them side by side, and the closed model's answers
become training rows.
"""
from __future__ import annotations

import json
from pathlib import Path

DATA = Path("data/tasks")

EMAIL_QUESTIONS = [
    ("spam", "Is this email spam or unsolicited bulk mail?", ["Yes", "No"], "noul"),
    ("reply", "Does this email need a reply from the recipient?", ["Yes", "No"], "noul"),
    ("priority", "How urgent is this email for the recipient?",
     ["high: needs action today", "medium: needs action this week", "low: no action or whenever"], "choice"),
    ("category", "Which category fits this email best?",
     ["work: business, projects, meetings, colleagues", "personal: friends, family, private matters",
      "promotion: advertising, offers, marketing", "notification: automatic system or service messages",
      "finance: invoices, payments, banking, trading", "other: anything else"], "choice"),
]


def load_items(task: str, count: int, offset: int = 0) -> list[dict]:
    file = DATA / f"{task}.jsonl"
    items = [json.loads(line) for line in open(file)]
    return items[offset : offset + count]


def questions_for(task: str, item: dict) -> list[tuple[str, str, list[str], str]]:
    """(name, question, options, kind) per question of this item."""
    if task == "emails":
        return EMAIL_QUESTIONS
    if task == "products":
        categories = json.loads((DATA / "products_categories.json").read_text())
        return [("category", "Which category does this product belong to?",
                 [c.replace("_", " ") for c in categories], "choice")]
    raise ValueError(task)


def gold_of(task: str, item: dict, name: str, options: list[str]) -> int | None:
    """Index of the gold option, where the dataset has one."""
    gold = item.get("gold") or {}
    if name not in gold:
        return None
    value = gold[name]
    if task == "products":
        wanted = str(value).replace("_", " ")
        return options.index(wanted) if wanted in options else None
    if name == "spam":
        return 0 if value else 1
    return None
