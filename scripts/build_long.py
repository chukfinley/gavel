#!/usr/bin/env python3
"""Long-state decisions.

The TypeSafe fixture has a median state of 2576 tokens; 91 of its 102 rows are
longer than 256 tokens. A model trained only on short pairs cannot use that
evidence — measured: accuracy 0.353 at 256 tokens, 0.471 at 1024, and 0.392 at
3072, because the training length was 224 and length does not extrapolate on
its own.

Three sources give long states:

* QuALITY — long articles with four options each.
* RACE — examination passages with four options each.
* Synthetic review packets — a JSON document with structured facts and one
  question about one line of it. This is the shape the TypeSafe invoice and
  incident rows use, and the label follows from the generated facts, thus it
  is exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typedec.schema import Decision, Option, write_jsonl  # noqa: E402


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


LINE_KINDS = [
    ("work_or_goods", "bills work performed or goods delivered"),
    ("tax", "is a tax amount"),
    ("charge_on_top", "is a charge added on top, such as freight or a fee"),
    ("credit_or_discount", "is a credit or a discount"),
    ("expense_or_retainage", "is a pass-through expense or retainage"),
]

LINE_TEXT = {
    "work_or_goods": ["Professional services, March", "Assembly labour, 12 hours",
                      "Steel brackets, 40 pieces", "Site supervision, week 14"],
    "tax": ["VAT 19 percent", "Sales tax, state", "Reverse charge VAT", "Value added tax 7 percent"],
    "charge_on_top": ["Freight and handling", "Fuel surcharge", "Small order fee", "Express delivery fee"],
    "credit_or_discount": ["Early payment discount", "Credit for returned goods",
                           "Volume rebate", "Goodwill credit"],
    "expense_or_retainage": ["Travel expenses, reimbursed at cost", "Retainage held, 5 percent",
                             "Permit fee paid on behalf of the customer", "Parking and tolls"],
}

INCIDENT_FIELDS = [
    ("affected_accounts", [1, 3, 12, 240, 5100]),
    ("workaround_available", [True, False]),
    ("data_exposed", [True, False]),
    ("service_reachable", [True, False]),
]


def packet_rows(count: int, rng: random.Random, filler_lines: int) -> list[Decision]:
    """Invoice review packets: a long JSON state, one typed question about one line."""
    out = []
    kinds = [kind for kind, _ in LINE_KINDS]
    for index in range(count):
        target_kind = rng.choice(kinds)
        target_text = rng.choice(LINE_TEXT[target_kind])
        lines, target_position = [], rng.randrange(1, filler_lines)
        for position in range(1, filler_lines + 1):
            if position == target_position:
                kind = target_kind
                text = target_text
            else:
                kind = rng.choice(kinds)
                text = rng.choice(LINE_TEXT[kind])
            lines.append({
                "line": position,
                "description": text,
                "quantity": rng.choice([1, 2, 5, 12, 40]),
                "unit_price": round(rng.uniform(10, 900), 2),
                "amount": round(rng.uniform(40, 9000), 2),
            })
        packet = {
            "task": "Accounts-payable review packet. Answer the question about the named line.",
            "vendor": {"name": f"Vendor {rng.randrange(100, 999)}",
                       "tax_id": f"DE{rng.randrange(100000000, 999999999)}",
                       "payment_terms": rng.choice(["net 30", "net 14", "net 60"])},
            "purchase_order": {"number": f"PO-{rng.randrange(10000, 99999)}",
                               "approved_total": round(rng.uniform(1000, 40000), 2),
                               "approver": rng.choice(["site manager", "head of operations"])},
            "delivery_evidence": [{"note": f"DN-{rng.randrange(1000, 9999)}",
                                   "received_by": rng.choice(["warehouse", "site office"]),
                                   "complete": rng.choice([True, False])} for _ in range(3)],
            "invoice": {"number": f"INV-{rng.randrange(10000, 99999)}",
                        "currency": "EUR", "lines": lines},
            "prior_invoices": [{"number": f"INV-{rng.randrange(10000, 99999)}",
                                "total": round(rng.uniform(500, 20000), 2)} for _ in range(4)],
        }
        options = [Option(kind, f'Line {target_position} ("{target_text}") {text}')
                   for kind, text in LINE_KINDS]
        rng.shuffle(options)
        label = [o.id for o in options].index(target_kind)
        out.append(Decision(
            id=rid("packet", str(index)),
            state=json.dumps(packet, indent=2),
            question=f'What is invoice line {target_position} ("{target_text}")? '
                     f"Pick the one kind that fits.",
            options=options, label=label, source="synth-packet", task="choice"))
    return out


def incident_rows(count: int, rng: random.Random) -> list[Decision]:
    """Incident packets: severity follows from the structured facts."""
    out = []
    for index in range(count):
        facts = {name: rng.choice(values) for name, values in INCIDENT_FIELDS}
        if facts["data_exposed"] or not facts["service_reachable"]:
            truth = 3
        elif facts["affected_accounts"] >= 240 and not facts["workaround_available"]:
            truth = 2
        elif facts["affected_accounts"] >= 12:
            truth = 1
        else:
            truth = 0
        packet = {
            "task": "Incident review. Set the severity from the facts.",
            "incident": {"id": f"INC-{rng.randrange(10000, 99999)}",
                         "reported_by": rng.choice(["on-call engineer", "support", "monitoring"]),
                         "component": rng.choice(["billing", "authentication", "reporting", "search"])},
            "facts": facts,
            "timeline": [{"minute": minute,
                          "note": rng.choice(["alert raised", "engineer acknowledged",
                                              "customer reported the same", "mitigation attempted",
                                              "traffic rerouted", "logs collected"])}
                         for minute in range(0, rng.choice([40, 90, 160]), 10)],
            "recent_changes": [{"service": rng.choice(["billing", "gateway", "database"]),
                                "change": rng.choice(["config update", "release", "schema migration"])}
                               for _ in range(3)],
        }
        names = ["low", "medium", "high", "critical"]
        out.append(Decision(
            id=rid("incident", str(index)), state=json.dumps(packet, indent=2),
            question="Which severity applies to this incident?",
            options=[Option(name, f"{name.capitalize()} severity") for name in names],
            label=truth, source="synth-incident", task="score"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/long.jsonl")
    parser.add_argument("--per-source", type=int, default=6000)
    parser.add_argument("--packets", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    rows: list[Decision] = []

    def report(name: str, produced: list[Decision]) -> None:
        rows.extend(produced)
        print(f"  {name}: {len(produced)}", flush=True)

    try:
        data = load_dataset("emozilla/quality", split="train")
        produced = []
        for index, row in enumerate(data):
            if len(produced) >= args.per_source:
                break
            options = row["options"]
            answer = int(row["answer"]) - 1 if int(row["answer"]) > 0 else 0
            if answer >= len(options):
                continue
            produced.append(Decision(
                id=rid("quality", str(index)), state=row["article"].strip(),
                question=row["question"].strip(),
                options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(options)],
                label=answer, source="quality", task="choice"))
        report("quality", produced)
    except Exception as error:                                   # noqa: BLE001
        print("  quality failed:", error, flush=True)

    try:
        data = load_dataset("ehovy/race", "all", split="train").shuffle(seed=args.seed)
        produced = []
        for index, row in enumerate(data):
            if len(produced) >= args.per_source:
                break
            letters = ["A", "B", "C", "D"]
            if row["answer"] not in letters:
                continue
            produced.append(Decision(
                id=rid("race", str(index)), state=row["article"].strip(),
                question=row["question"].strip(),
                options=[Option(f"o{i}", str(t).strip()) for i, t in enumerate(row["options"])],
                label=letters.index(row["answer"]), source="race", task="choice"))
        report("race", produced)
    except Exception as error:                                   # noqa: BLE001
        print("  race failed:", error, flush=True)

    report("synth-packet", packet_rows(args.packets, rng, filler_lines=8))
    report("synth-incident", incident_rows(args.packets // 2, rng))

    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
