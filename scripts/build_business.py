#!/usr/bin/env python3
"""Add business-process decisions, the domain the first run had never seen.

The TypeSafe fixture holds security incidents, invoice handling, support
routing and agent traces. The first training mix (inference, school knowledge,
sentiment) contains none of that, and the model scored 0.569 there while it
already beat the baseline on WANLI.

Two sources fill the gap:

* Public label sets with the same *shape* — route to a topic, gate on risk,
  classify a message. Option sets are sampled for each row, thus the model
  reads the option text instead of learning a class index.
* Templates that build process decisions directly: queue routing, severity
  triage, escalation, and refund handling. The label follows from the facts
  written into the state, thus no teacher model is needed and no teacher noise
  enters the data.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedec.schema import Decision, Option, write_jsonl  # noqa: E402


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def sampled_options(names: list[str], truth: int, count: int, rng: random.Random):
    others = [i for i in range(len(names)) if i != truth]
    picked = rng.sample(others, min(count - 1, len(others))) + [truth]
    rng.shuffle(picked)
    return picked, picked.index(truth)


# ----------------------------------------------------------------- templates

QUEUES = {
    "billing": ["was charged twice for the same subscription period",
                "asks why the invoice total changed after the trial ended",
                "wants the receipt reissued with the company VAT number"],
    "account access": ["cannot sign in after resetting the password",
                       "lost the second factor device and is locked out",
                       "reports that the login link expires before it arrives"],
    "technical fault": ["reports that exports fail with a server error",
                        "says the mobile application closes when opening a report",
                        "sees empty charts after the last release"],
    "cancellation": ["wants to end the contract at the next renewal date",
                     "asks how to close the workspace permanently",
                     "requests deletion of the account and all data"],
    "sales": ["asks for a quote for fifty additional seats",
              "wants a comparison of the team and enterprise plans",
              "asks whether a yearly contract lowers the seat price"],
}

SEVERITY = [
    ("low", ["one user reports a cosmetic layout problem on one page",
             "a translation string is missing in a rarely used dialog"]),
    ("medium", ["a report export fails for some accounts, a manual export works",
                "login is slow for one region, but it completes"]),
    ("high", ["paying customers cannot sign in, no workaround exists",
              "the billing job charged a group of accounts twice"]),
    ("critical", ["customer data from one account is visible to another account",
                  "the production database is unreachable and the service is down"]),
]

ACTIONS = [
    ("approve", "the request is inside the stated limit and the evidence is complete"),
    ("hold for review", "the evidence is incomplete and one figure cannot be checked"),
    ("reject", "the request breaks a stated limit"),
    ("escalate to a human", "the case is outside the written policy"),
]

REFUND_FACTS = [
    ("refund in full", "The purchase is nine days old and the plan allows a full refund inside fourteen days."),
    ("refund in part", "The purchase is forty days old; the policy allows a pro-rata refund after thirty days."),
    ("no refund", "The purchase is eight months old and the policy allows refunds inside fourteen days only."),
    ("escalate to a human", "The purchase date is missing from the record, thus the policy cannot be applied."),
]


def routing_rows(count: int, rng: random.Random) -> list[Decision]:
    names = list(QUEUES)
    out = []
    for index in range(count):
        truth = rng.randrange(len(names))
        picked, label = sampled_options(names, truth, rng.choice([3, 4, 5]), rng)
        body = rng.choice(QUEUES[names[truth]])
        out.append(Decision(
            id=rid("route", str(index)),
            state=f"A customer writes to support. The customer {body}.",
            question="Which queue should handle this request?",
            options=[Option(names[i], f"{names[i].capitalize()} support") for i in picked],
            label=label, source="synth-routing", task="choice"))
    return out


def severity_rows(count: int, rng: random.Random) -> list[Decision]:
    names = [name for name, _ in SEVERITY]
    out = []
    for index in range(count):
        truth = rng.randrange(len(SEVERITY))
        body = rng.choice(SEVERITY[truth][1])
        out.append(Decision(
            id=rid("severity", str(index)),
            state=f"An incident report arrives: {body}.",
            question="Which severity applies to this incident?",
            options=[Option(name, f"{name.capitalize()} severity") for name in names],
            label=truth, source="synth-severity", task="score"))
    return out


def action_rows(count: int, rng: random.Random) -> list[Decision]:
    names = [name for name, _ in ACTIONS]
    out = []
    for index in range(count):
        truth = rng.randrange(len(ACTIONS))
        reason = ACTIONS[truth][1]
        amount = rng.choice([120, 480, 1500, 5400])
        out.append(Decision(
            id=rid("action", str(index)),
            state=(f"An expense claim for {amount} euro is checked against the policy. "
                   f"The reviewer notes that {reason}."),
            question="Which action should the system take?",
            options=[Option(name, name.capitalize()) for name in names],
            label=truth, source="synth-action", task="choice"))
    return out


def refund_rows(count: int, rng: random.Random) -> list[Decision]:
    names = [name for name, _ in REFUND_FACTS]
    out = []
    for index in range(count):
        truth = rng.randrange(len(REFUND_FACTS))
        out.append(Decision(
            id=rid("refund", str(index)),
            state=REFUND_FACTS[truth][1],
            question="Which refund decision follows from the policy?",
            options=[Option(name, name.capitalize()) for name in names],
            label=truth, source="synth-refund", task="choice"))
    return out


def evidence_rows(count: int, rng: random.Random) -> list[Decision]:
    """Evidence judgments, the shape of the authored fixture."""
    claims = [
        ("The order has shipped.", "The warehouse confirms the parcel left the building this morning.", 0),
        ("The order has shipped.", "The warehouse says the parcel is still waiting for collection.", 2),
        ("The order has shipped.", "The warehouse has not answered the request yet.", 1),
        ("The invoice is paid.", "The bank statement shows the transfer cleared on Monday.", 0),
        ("The invoice is paid.", "The bank returned the transfer for a wrong account number.", 2),
        ("The invoice is paid.", "The finance team is still reconciling last week's transfers.", 1),
        ("The contract is signed by both parties.", "Both signature blocks carry a date and a name.", 0),
        ("The contract is signed by both parties.", "Only the supplier signed; the customer block is empty.", 2),
        ("The contract is signed by both parties.", "The signature page was not included in the scan.", 1),
    ]
    options = [Option("supported", "The evidence establishes the claim"),
               Option("insufficient", "The evidence does not establish either"),
               Option("contradicted", "The evidence establishes the opposite")]
    out = []
    for index in range(count):
        claim, evidence, label = rng.choice(claims)
        out.append(Decision(
            id=rid("evidence", str(index), claim, evidence),
            state=evidence,
            question=f"Assess the claim: {claim.rstrip('.')}",
            options=list(options), label=label,
            source="synth-evidence", task="choice"))
    return out


# ------------------------------------------------------------ public sources

def public_rows(per_source: int, rng: random.Random) -> list[Decision]:
    from datasets import load_dataset
    out: list[Decision] = []

    def report(name: str, rows: list[Decision]) -> None:
        out.extend(rows)
        print(f"  {name}: {len(rows)}", flush=True)

    def topic(name: str, dataset, text_key: str, names: list[str], label_key: str,
              question: str, source: str) -> list[Decision]:
        rows = []
        for index, row in enumerate(dataset):
            if len(rows) >= per_source:
                break
            truth = int(row[label_key])
            picked, label = sampled_options(names, truth, rng.choice([3, 4, 5]), rng)
            rows.append(Decision(
                id=rid(source, str(index)),
                state=str(row[text_key]).strip()[:1200],
                question=question,
                options=[Option(names[i], names[i]) for i in picked],
                label=label, source=source, task="choice"))
        return rows

    try:
        data = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=1)
        report("ag_news", topic("ag_news", data, "text",
                                ["World news", "Sports", "Business", "Science and technology"],
                                "label", "Which topic does this article belong to?", "ag-news"))
    except Exception as error:                                   # noqa: BLE001
        print("  ag_news failed:", error, flush=True)

    try:
        data = load_dataset("fancyzhx/dbpedia_14", split="train").shuffle(seed=1)
        names = ["Company", "Educational institution", "Artist", "Athlete", "Office holder",
                 "Means of transport", "Building", "Natural place", "Village", "Animal",
                 "Plant", "Album", "Film", "Written work"]
        report("dbpedia", topic("dbpedia", data, "content", names, "label",
                                "Which category does this description belong to?", "dbpedia"))
    except Exception as error:                                   # noqa: BLE001
        print("  dbpedia failed:", error, flush=True)

    for config, question, positive, negative in [
        ("offensive", "Does this message contain offensive language?", "Yes, it is offensive", "No, it is not offensive"),
        ("hate", "Does this message attack a group of people?", "Yes, it attacks a group", "No, it does not"),
    ]:
        try:
            data = load_dataset("cardiffnlp/tweet_eval", config, split="train").shuffle(seed=1)
            rows = []
            for index, row in enumerate(data):
                if len(rows) >= per_source // 2:
                    break
                rows.append(Decision(
                    id=rid(f"tweet-{config}", str(index)),
                    state=row["text"].strip(), question=question,
                    options=[Option("yes", positive), Option("no", negative)],
                    label=0 if int(row["label"]) == 1 else 1,
                    source=f"tweet-{config}", task="noul"))
            report(f"tweet-{config}", rows)
        except Exception as error:                               # noqa: BLE001
            print(f"  tweet-{config} failed:", error, flush=True)

    try:
        data = load_dataset("ucirvine/sms_spam", split="train").shuffle(seed=1)
        rows = []
        for index, row in enumerate(data):
            if len(rows) >= per_source // 2:
                break
            rows.append(Decision(
                id=rid("sms-spam", str(index)), state=row["sms"].strip(),
                question="Should this message be blocked as unsolicited advertising?",
                options=[Option("yes", "Yes, block it"), Option("no", "No, deliver it")],
                label=0 if int(row["label"]) == 1 else 1,
                source="sms-spam", task="noul"))
        report("sms-spam", rows)
    except Exception as error:                                   # noqa: BLE001
        print("  sms_spam failed:", error, flush=True)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/business.jsonl")
    parser.add_argument("--per-source", type=int, default=12000)
    parser.add_argument("--synthetic", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[Decision] = []
    print("public sources", flush=True)
    rows.extend(public_rows(args.per_source, rng))
    print("templates", flush=True)
    for name, builder in [("routing", routing_rows), ("severity", severity_rows),
                          ("action", action_rows), ("refund", refund_rows),
                          ("evidence", evidence_rows)]:
        produced = builder(args.synthetic // 5, rng)
        rows.extend(produced)
        print(f"  synth-{name}: {len(produced)}", flush=True)

    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
