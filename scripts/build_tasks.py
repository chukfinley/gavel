#!/usr/bin/env python3
"""Items for the fixed-question tasks, and the same data as training sources.

Emails: SetFit/enron_spam (real mail, spam gold). Products:
Lezh1n/ecommerce-product-classification-by-categories (32 categories).
The task files hold a balanced sample from the held-out splits for the
WebUI; the training sources come from the train splits.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

TOKEN = os.environ.get("HF_TOKEN")
OUT = Path("data/tasks")
OUT.mkdir(parents=True, exist_ok=True)


def email_state(row: dict) -> str:
    subject = (row.get("subject") or "").strip()
    body = (row.get("message") or row.get("text") or "").strip()
    return f"Subject: {subject}\n\n{body[:2500]}"


def main() -> None:
    rng = random.Random(11)
    # ---- emails
    train = [json.loads(l) for l in open(hf_hub_download("SetFit/enron_spam", "train.jsonl", repo_type="dataset",
                                                            local_dir="_scratch/hf/enron", token=TOKEN))]
    test = [json.loads(l) for l in open(hf_hub_download("SetFit/enron_spam", "test.jsonl", repo_type="dataset",
                                                           local_dir="_scratch/hf/enron", token=TOKEN))]
    rng.shuffle(test)
    spam = [r for r in test if r["label_text"] == "spam"][:200]
    ham = [r for r in test if r["label_text"] == "ham"][:200]
    items = spam + ham
    rng.shuffle(items)
    with open(OUT / "emails.jsonl", "w") as handle:
        handle.writelines(json.dumps({"id": f"enron-{r['message_id']}", "state": email_state(r),
                                      "gold": {"spam": r["label_text"] == "spam"}}) + "\n" for r in items)
    rng.shuffle(train)
    with open("data/emails_spam.jsonl", "w") as handle:
        for r in train[:8000]:
            state = email_state(r)
            handle.write(json.dumps({
                "id": hashlib.sha1(f"enron|{r['message_id']}".encode()).hexdigest()[:20],
                "state": state, "question": "Is this email spam or unsolicited bulk mail?",
                "options": [{"id": "Yes", "description": "Yes"}, {"id": "No", "description": "No"}],
                "label": 0 if r["label_text"] == "spam" else 1, "source": "enron-spam", "task": "noul"}) + "\n")
    print(f"emails: {len(items)} task items, 8000 training rows")
    # ---- products
    repo = "Lezh1n/ecommerce-product-classification-by-categories"
    frames = {split: pd.read_parquet(hf_hub_download(repo, f"data/{split}-00000-of-00001.parquet", repo_type="dataset",
                                                      local_dir="_scratch/hf/products", token=TOKEN))
              for split in ("train", "test")}
    categories = sorted(frames["train"]["label"].unique().tolist())
    (OUT / "products_categories.json").write_text(json.dumps(categories, indent=1))
    test = frames["test"].sample(frac=1.0, random_state=11)
    per_category = 13
    picked = test.groupby("label").head(per_category)
    with open(OUT / "products.jsonl", "w") as handle:
        handle.writelines(json.dumps({"id": f"prod-{int(r.id)}", "state": str(r.text)[:1500],
                                      "gold": {"category": r.label}}) + "\n" for r in picked.itertuples())
    options = [{"id": c, "description": c.replace("_", " ")} for c in categories]
    with open("data/products.jsonl", "w") as handle:
        for r in frames["train"].itertuples():
            if r.label not in categories:
                continue
            handle.write(json.dumps({
                "id": hashlib.sha1(f"product|{int(r.id)}".encode()).hexdigest()[:20],
                "state": str(r.text)[:1500], "question": "Which category does this product belong to?",
                "options": options, "label": categories.index(r.label), "source": "products", "task": "choice"}) + "\n")
    print(f"products: {len(picked)} task items over {len(categories)} categories, {len(frames['train'])} training rows")


if __name__ == "__main__":
    main()
