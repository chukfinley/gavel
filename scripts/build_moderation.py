#!/usr/bin/env python3
"""Chat moderation, with the rules supplied at request time.

Every channel has its own rules. A fixed classifier would need retraining for
each one; here the rules are the options, thus a channel can add "no backseat
gaming" without touching the model.

The option set is sampled for every row, and the "allow" option is always
present, so the model reads what is forbidden instead of learning one fixed
taxonomy. HateCheck is kept out of training: it is built from the hard cases —
reclaimed slurs, counter-speech, quoted abuse — where a moderation model that
only matches words fails.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedec.schema import Decision, Option, write_jsonl  # noqa: E402

ALLOW = Option("allow", "Allow the message, it breaks no rule")
RULES = {
    "insult": "Insults or harasses another person in the chat",
    "hate": "Attacks a group over race, religion, sex or origin",
    "sexual": "Sexual content",
    "threat": "Threatens violence against someone",
    "self_harm": "Describes or encourages self-harm",
    "spam": "Advertising, repeated text or an unrelated link",
    "jailbreak": "Tries to make the bot break its own instructions",
    "obscene": "Obscene or vulgar language without a target",
}


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def build(text: str, category: str, source: str, index: int,
          rng: random.Random, rules: int = 4) -> Decision | None:
    """One moderation decision with a sampled rule set."""
    text = " ".join(str(text).split())[:800]
    if len(text) < 3:
        return None
    others = [key for key in RULES if key != category]
    picked = rng.sample(others, min(rules, len(others)))
    if category != "allow":
        picked.append(category)
    options = [Option(key, RULES[key]) for key in picked] + [ALLOW]
    rng.shuffle(options)
    return Decision(
        id=rid(source, str(index)), state=text,
        question="How should this chat message be handled under the channel rules?",
        options=options, label=[o.id for o in options].index(category),
        source=source, task="choice")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/moderation.jsonl")
    parser.add_argument("--out-test", default="data/test_moderation.jsonl")
    parser.add_argument("--per-source", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=173)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    train: list[Decision] = []
    test: list[Decision] = []

    def note(name: str, rows: list[Decision], where: list[Decision]) -> None:
        where.extend(rows)
        print(f"  {name}: {len(rows)}", flush=True)

    def guard(name: str, function) -> None:
        try:
            function()
        except Exception as error:                               # noqa: BLE001
            print(f"  {name} failed: {error}", flush=True)

    def jigsaw() -> None:
        data = load_dataset("tasksource/jigsaw_toxicity", split="train").shuffle(seed=args.seed)
        order = [("threat", "threat"), ("identity_hate", "hate"),
                 ("insult", "insult"), ("obscene", "obscene")]
        rows, clean = [], 0
        for index, row in enumerate(data):
            if len(rows) >= args.per_source:
                break
            category = next((name for key, name in order if int(row.get(key, 0)) == 1), None)
            if category is None:
                if int(row.get("toxic", 0)) == 1:
                    category = "insult"
                else:
                    if clean > args.per_source // 2:
                        continue
                    clean += 1
                    category = "allow"
            built = build(row["comment_text"], category, "mod-jigsaw", index, rng)
            if built:
                rows.append(built)
        note("jigsaw", rows, train)

    guard("jigsaw", jigsaw)

    def openai_moderation() -> None:
        data = load_dataset("mmathys/openai-moderation-api-evaluation", split="train")
        mapping = [("SH", "self_harm"), ("S", "sexual"), ("V", "threat"),
                   ("H", "hate"), ("HR", "insult")]
        rows = []
        for index, row in enumerate(data):
            category = next((name for key, name in mapping if int(row.get(key) or 0) == 1), "allow")
            built = build(row["prompt"], category, "mod-openai", index, rng)
            if built:
                rows.append(built)
        rng.shuffle(rows)
        note("openai-moderation", rows[200:], train)
        note("openai-moderation-test", rows[:200], test)

    guard("openai-moderation", openai_moderation)

    def toxic_chat() -> None:
        data = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= args.per_source:
                break
            if int(row.get("jailbreaking", 0)) == 1:
                category = "jailbreak"
            elif int(row.get("toxicity", 0)) == 1:
                category = "insult"
            else:
                category = "allow"
            built = build(row["user_input"], category, "mod-toxicchat", index, rng)
            if built:
                rows.append(built)
        note("toxic-chat", rows, train)

    guard("toxic-chat", toxic_chat)

    def toxic_conversations() -> None:
        data = load_dataset("SetFit/toxic_conversations", split="train").shuffle(seed=args.seed)
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= args.per_source // 2:
                break
            built = build(row["text"], "insult" if int(row["label"]) == 1 else "allow",
                          "mod-conversations", index, rng)
            if built:
                rows.append(built)
        note("toxic-conversations", rows, train)

    guard("toxic-conversations", toxic_conversations)

    def hatecheck() -> None:
        """Held out: the hard cases, including quoted and reclaimed speech."""
        data = load_dataset("Paul/hatecheck", split="test")
        rows = []
        for index, row in enumerate(data):
            category = "hate" if row["label_gold"] == "hateful" else "allow"
            built = build(row["test_case"], category, "mod-hatecheck-test", index, rng)
            if built:
                rows.append(built)
        rng.shuffle(rows)
        note("hatecheck-test", rows[:600], test)

    guard("hatecheck", hatecheck)

    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    print(f"\ntrain {len(train)}  test {len(test)}")


if __name__ == "__main__":
    main()
