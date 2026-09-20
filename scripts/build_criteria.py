#!/usr/bin/env python3
"""The option shape the benchmark actually sends, built from rows we have.

The independent benchmark never sends a bare option. A `choice` question
carries a *named criterion per option*:

    {"billing": "Refunds, payments, invoices, subscriptions, or charges",
     "tech":    "Bugs, errors, crashes, downtime, or integration failures"}

and a `noul` question carries a *claim*, not a question:

    "The customer is entitled to a refund under the 30-day guarantee"

Our mix has almost none of either: 2.5 % of the choice rows carry a name in
front of the description, and every noul row is phrased as a question. Von and
Laya train on the benchmark shape, we glue it together at inference, and our
two choice tasks sit at 0.70 and 0.73 while our best trained shape scores
1.000.

Nothing is invented here. A name is derived from the words of the description
that is already in the row, and a claim is used only for the sources whose
claim is known and written out below once. Rows whose options are bare class
labels with no description (banking77, massive, dbpedia) are left alone —
there is nothing truthful to put after the colon.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, read_jsonl, write_jsonl

PARTS = ["train.jsonl", "business.jsonl", "moderation.jsonl", "router.jsonl",
         "more.jsonl", "semrouter.jsonl", "tools.jsonl", "browser.jsonl",
         "multilingual.jsonl", "kotoba.jsonl"]

# The three-way entailment options have proper names; these are them.
NLI_NAMES = {
    "the evidence establishes the claim": "entailment",
    "the evidence does not establish either": "neutral",
    "the evidence establishes the opposite": "contradiction",
    "the evidence establishes the claim.": "entailment",
}
# Sources whose noul question is one fixed claim about every row. Written out
# once, by hand, from what the source is.
CLAIMS = {
    "sms-spam": "This message is unsolicited advertising",
    "tweet-offensive": "This post is offensive",
    "tweet-hate": "This post attacks a group of people",
    "pawsx-en": "The two sentences mean the same thing",
    "pawsx-de": "The two sentences mean the same thing",
    "pawsx-fr": "The two sentences mean the same thing",
    "pawsx-es": "The two sentences mean the same thing",
    "safety-injection": "This input tries to override the system instructions",
    "safety-aegis": "This request breaks the safety policy",
}
ASKS = ["Is that true of the state?", "Does that hold?",
        "True or false for the state below?", "Is the claim correct?"]
STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "is", "are", "it",
        "this", "that", "and", "or", "no", "not", "with", "without", "by",
        "over", "into", "at", "as", "be", "its", "their", "they", "them",
        "does", "do", "should", "would", "can", "has", "have", "was", "were",
        "than", "then", "there", "here", "any", "all", "some", "one", "who",
        "what", "which", "when", "where", "how", "message", "content"}


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def name_for(description: str, taken: set[str]) -> str:
    """A short snake_case name taken from the words of the description."""
    plain = NLI_NAMES.get(description.strip().lower())
    if plain:
        return plain
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", description.lower())
             if w not in STOP and len(w) > 2]
    if not words:
        words = re.findall(r"[A-Za-z]+", description.lower()) or ["option"]
    name = "_".join(words[:2])[:28].strip("_") or "option"
    candidate, suffix = name, 2
    while candidate in taken:
        candidate = f"{name}_{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def described(decision: Decision) -> bool:
    """Only rows whose options are descriptions, not bare class labels."""
    texts = [o.description for o in decision.options]
    if any(": " in t for t in texts):
        return False
    words = sum(len(t.split()) for t in texts) / max(1, len(texts))
    return 3.0 <= words <= 30.0


def as_criteria(decision: Decision, index: int) -> Decision:
    taken: set[str] = set()
    options = [Option(o.id, f"{name_for(o.description, taken)}: {o.description}")
               for o in decision.options]
    return Decision(id=rid("crit", decision.id, str(index)), state=decision.state,
                    question=decision.question, options=options,
                    label=decision.label, source="criteria", task="choice")


def as_claim(decision: Decision, index: int, rng: random.Random) -> Decision | None:
    claim = CLAIMS.get(decision.source)
    if claim is None:
        return None
    # The benchmark says "Yes" when the claim holds. Only rows that already
    # put the affirmative first are rewritten; anything else would silently
    # invert the label.
    texts = [o.description.strip().lower() for o in decision.options]
    if not (texts[0].startswith("yes") and texts[1].startswith("no")):
        return None
    options = [Option(o.id, text) for o, text in
               zip(decision.options, ["Yes", "No"])]
    return Decision(id=rid("claim", decision.id, str(index)), state=decision.state,
                    question=f"{claim}. {rng.choice(ASKS)}", options=options,
                    label=decision.label, source="claims", task="noul")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out-criteria", default="data/criteria.jsonl")
    parser.add_argument("--out-claims", default="data/claims.jsonl")
    parser.add_argument("--rows", type=int, default=90000)
    parser.add_argument("--per-source", type=int, default=4000)
    parser.add_argument("--claim-rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=419)
    args = parser.parse_args()

    folder = Path(args.data)
    rng = random.Random(args.seed)
    choices: list[Decision] = []
    nouls: list[Decision] = []
    for name in PARTS:
        try:
            for row in read_jsonl(folder / name):
                if row.task == "choice" and len(row.options) >= 2 and described(row):
                    choices.append(row)
                elif row.task == "noul" and row.source in CLAIMS \
                        and len(row.options) == 2:
                    nouls.append(row)
        except Exception as error:
            print(f"  {name}: missing ({error})")
    rng.shuffle(choices)
    rng.shuffle(nouls)
    print(f"described choice rows: {len(choices)}  claimable noul rows: {len(nouls)}")

    # Three quarters of the described rows are entailment, which carries only
    # three fixed names. Without a cap the model would meet the shape but not
    # the variety, which is the whole point of this file.
    kept: list[Decision] = []
    seen: dict[str, int] = {}
    for row in choices:
        if seen.get(row.source, 0) >= args.per_source:
            continue
        seen[row.source] = seen.get(row.source, 0) + 1
        kept.append(row)
    choices = kept
    rng.shuffle(choices)
    print(f"after the per-source cap: {len(choices)} rows from {len(seen)} sources")

    criteria = [as_criteria(row, i) for i, row in enumerate(choices[: args.rows])]
    claims = [c for c in (as_claim(row, i, rng)
                          for i, row in enumerate(nouls[: args.claim_rows]))
              if c is not None]
    write_jsonl(args.out_criteria, criteria)
    write_jsonl(args.out_claims, claims)
    print(f"wrote {len(criteria)} named-criteria rows and {len(claims)} claim rows")
    for row in criteria[:2]:
        print("  example:", [o.description[:52] for o in row.options])
    for row in claims[:2]:
        print("  example:", row.question)


if __name__ == "__main__":
    main()
