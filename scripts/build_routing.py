#!/usr/bin/env python3
"""Specialist routing, where a rule in the question decides the edge cases.

Our worst family on JevBench's standard tier is routing: 2 of 12. Looking at
those items shows why. They are short requests over six named specialists,
and the hard part is not the topic — it is a rule carried in the
instructions, of the shape "file edits with test execution use the agent,
even if it is code-related". Two categories overlap on purpose and a
sentence decides which wins.

Our mix has routing, but the wrong kind: `router-tier` and `semrouter` pick a
model size or a difficulty band. Nothing in it teaches "these two categories
overlap, and here is the tie-break".

**This builds the shape, not the cases.** The categories, the requests and
the tie-break rules here are written for this file. Generating a benchmark's
own test items is what makes Von's 0.923 meaningless, and it is not done
here: no string is taken from any suite. What transfers is the form — named
criteria, overlapping categories, and a rule in the instruction that decides
between them.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl

# Each family: the categories with their criteria, the requests per category,
# and a pair of categories that overlap with the rule that separates them.
FAMILIES = [
    {
        "question": "Which specialist should take this request?",
        "criteria": {
            "calculation": "A self-contained sum, proof or numeric problem",
            "scripting": "Writing or explaining code that stands on its own",
            "repository": "Reading or changing files in a project, or running its tests",
            "lookup": "Answering from a document that was supplied",
            "service": "Carrying out an action in an external system",
            "anything else": "None of the specialist categories fit",
        },
        "requests": {
            "calculation": [
                "What is the greatest common divisor of 84 and 126?",
                "Work out the compound interest on 4000 at 3 percent over 7 years.",
                "Show that the sum of two odd numbers is even.",
                "Convert 12 stone 4 pounds into kilograms.",
                "How many distinct ways can eight people sit around a round table?",
                "Estimate the surface area of a cylinder 30 cm tall with radius 8 cm.",
            ],
            "scripting": [
                "Write a function that removes duplicates from a list but keeps the order.",
                "Explain what a generator expression does compared with a list comprehension.",
                "Give me a regular expression that matches a UK postcode.",
                "Show how to parse an ISO timestamp into a datetime.",
                "Write a small class that caches the results of a slow function.",
            ],
            "repository": [
                "Open the settings module and change the default timeout to thirty seconds.",
                "Run the test suite and tell me which cases fail.",
                "Find every place that still imports the deprecated client and update them.",
                "Add a migration for the new column and apply it.",
                "Rename the helper in every file that uses it, then run the tests.",
            ],
            "lookup": [
                "According to the attached handbook, how much notice does a tenant owe?",
                "From the report above, what was revenue in the second quarter?",
                "The specification is attached. What is the maximum payload size?",
                "Using the supplied minutes, who was assigned the budget review?",
            ],
            "service": [
                "Book a meeting room for Thursday at ten and invite the design team.",
                "Refund the duplicate charge on invoice 4471.",
                "Send the signed contract to the address on file.",
                "Cancel the subscription at the end of the billing period.",
            ],
            "anything else": [
                "Good morning, hope your week is going well.",
                "What do you think about the weather lately?",
                "Thanks, that was helpful.",
            ],
        },
        "rules": [
            (("Anything that touches files in a project, or runs its tests, goes "
             "to repository even when it is also a coding question."),
             [("Write a caching decorator and add it to utils.py, then run the tests.",
               "repository"),
              ("Explain how a caching decorator works.", "scripting"),
              ("Fix the failing test in test_parser.py.", "repository"),
              ("Write a parser for this grammar; nothing needs saving.", "scripting")]),
            (("A question that can be answered from a document that was supplied "
             "goes to lookup, even when the subject is numeric or technical."),
             [("The attached invoice lists three line items. What do they add up to?",
               "lookup"),
              ("What do 249, 1130 and 88 add up to?", "calculation"),
              ("From the attached manual, what voltage does the unit need?", "lookup")]),
        ],
    },
    {
        "question": "Which queue should this ticket go to?",
        "criteria": {
            "payments": "Charges, invoices, refunds and subscriptions",
            "access": "Signing in, passwords, permissions and account data",
            "faults": "Something is broken, erroring or unavailable",
            "onboarding": "Setting up, importing data or a first configuration",
            "commercial": "Prices, quotes, renewals and new purchases",
            "everything else": "Does not belong to any of the other queues",
        },
        "requests": {
            "payments": [
                "We were billed twice for the same month, please sort it out.",
                "The invoice has the wrong company address on it.",
                "Can we move from monthly to yearly billing?",
                "A refund was promised three weeks ago and has not arrived.",
            ],
            "access": [
                "My colleague left and I need her account closed.",
                "The password reset mail never arrives.",
                "I need admin rights on the reporting workspace.",
                "Please export everything you hold about me and then delete it.",
            ],
            "faults": [
                "Uploads over ten megabytes fail with a server error.",
                "The dashboard has been blank since this morning.",
                "Notifications arrive hours late, if at all.",
                "The mobile app closes itself as soon as I open a report.",
            ],
            "onboarding": [
                "We are starting next week and need our users imported from a CSV.",
                "How do we connect our identity provider for the first time?",
                "What is the recommended setup for a team of fifty?",
            ],
            "commercial": [
                "What would twenty extra seats cost on our current plan?",
                "We need a quote for a three year term with support included.",
                "Our renewal is in March, can we talk about the terms?",
            ],
            "everything else": [
                "Just wanted to say the new release is a big improvement.",
                "Is your office open between Christmas and New Year?",
            ],
        },
        "rules": [
            (("A request about money that is blocked by something broken goes to "
             "faults first, because the fault has to be fixed before the money "
             "can be corrected."),
             [("Checkout throws an error so nobody can pay us this morning.", "faults"),
              ("The card was charged twice, everything else works.", "payments"),
              ("Invoices stopped generating after your update last night.", "faults")]),
            (("A permission problem during a first setup belongs to onboarding, "
             "not access."),
             [(("We are configuring the account this week and the admin invite is "
               "rejected."), "onboarding"),
              ("I have used this for a year and today my login stopped working.",
               "access")]),
        ],
    },
]
PHRASINGS = ["{q}", "{q} Pick exactly one.", "Route this. {q}",
             "{q} Choose the single best fit."]


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def make(rng, state, gold, criteria, question, source) -> Decision:
    keys = list(criteria)
    rng.shuffle(keys)
    options = [Option(k, f"{k}: {criteria[k]}") for k in keys]
    return Decision(id=rid("route", state, gold, question),
                    state=state, question=question, options=options,
                    label=keys.index(gold), source=source, task="choice")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/routing.jsonl")
    parser.add_argument("--out-test", default="data/test_routing.jsonl")
    parser.add_argument("--rows", type=int, default=24000)
    parser.add_argument("--test-rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=727)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out: list[Decision] = []
    while len(out) < args.rows + args.test_rows:
        family = rng.choice(FAMILIES)
        criteria = dict(family["criteria"])
        # Half the rows drop a category the answer does not need, so the label
        # set is not always the same six strings.
        base = rng.choice(PHRASINGS).format(q=family["question"])

        if rng.random() < 0.45 and family["rules"]:
            rule, cases = rng.choice(family["rules"])
            state, gold = rng.choice(cases)
            question = f"{base} {rule}"
            source = "routing-rule"
        else:
            gold = rng.choice(list(family["requests"]))
            state = rng.choice(family["requests"][gold])
            question = base
            source = "routing-specialist"

        if rng.random() < 0.5:
            droppable = [k for k in criteria if k != gold]
            for key in rng.sample(droppable, k=min(2, len(droppable) - 1)):
                criteria.pop(key)
        out.append(make(rng, state, gold, criteria, question, source))

    rng.shuffle(out)
    write_jsonl(args.out_test, out[: args.test_rows])
    write_jsonl(args.out_train, out[args.test_rows :])
    print(f"wrote {len(out) - args.test_rows} training and "
          f"{args.test_rows} test routing rows")


if __name__ == "__main__":
    main()
