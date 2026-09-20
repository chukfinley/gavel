#!/usr/bin/env python3
"""The domains the benchmarks test and our mix never had.

v2 of `jabr/classifier-benchmark` is 49 tasks and 869 cases, and every open
model falls over on it — Von loses 25.7 points, GLiNER2 10.7. It is not
harder than v1, it is *elsewhere*: grammar, commit messages, code review,
PII, phishing, formality, reading level, spoilers, allergens, contract
clauses, insurance, delivery. JevBench's standard tier adds answer adequacy
and routing, where the only encoder of our size on that board scores 0.431.

Every source below was probed on 2026-09-20 and loads without a dataset
script. Sources that need one (TREC, PIQA, social_i_qa, CUAD, commitpackft,
phishing-dataset) are gone from the Hub's supported path and are not used.

Each source becomes typed decisions in our own schema: `choice` with named
criteria where the label set has meaning, `noul` as a claim, `score` for
anything ordered.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.schema import Decision, Option, write_jsonl

QUESTION_STYLES = ["{q}", "{q} Choose one.", "Decide: {q}", "{q} Pick the best fit."]


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def options_from(criteria: dict[str, str], rng, shuffle=True) -> list[Option]:
    """Named criteria, the shape the benchmarks send."""
    keys = list(criteria)
    if shuffle:
        rng.shuffle(keys)
    return [Option(k, f"{k}: {criteria[k]}" if criteria[k] else k) for k in keys]


def pick(rng, question: str) -> str:
    return rng.choice(QUESTION_STYLES).format(q=question)


# --------------------------------------------------------------- the sources
def grammar(rng, limit):
    """CoLA: is this sentence grammatical? The worst cell on v2 for everyone."""
    from datasets import load_dataset

    criteria = {"acceptable": "The sentence is well formed English",
                "unacceptable": "The sentence breaks a rule of English grammar"}
    out = []
    for row in load_dataset("nyu-mll/glue", "cola", split="train").select(
            range(min(limit, 8551))):
        options = options_from(criteria, rng)
        label = "acceptable" if row["label"] == 1 else "unacceptable"
        out.append(Decision(
            id=rid("cola", row["sentence"]), state=row["sentence"],
            question=pick(rng, "Is this sentence grammatically correct?"),
            options=options, label=[o.id for o in options].index(label),
            source="cola-grammar", task="choice"))
    return out


def pii(rng, limit):
    """Does this text contain personal data? A v2 noul task, 0.765 at best."""
    from datasets import load_dataset

    out = []
    stream = load_dataset("ai4privacy/pii-masking-200k", split="train",
                          streaming=True)
    for index, row in enumerate(stream):
        if len(out) >= limit:
            break
        text = row["source_text"]
        has = bool(row.get("privacy_mask"))
        # Rows without a mask are rare here, so half the negatives are made by
        # taking a masked row's already-redacted target text.
        if has and index % 2 == 1:
            text, has = row["target_text"], False
        options = [Option("Yes", "Yes"), Option("No", "No")]
        out.append(Decision(
            id=rid("pii", text[:120], str(has)), state=text,
            question="This text contains personal data about an identifiable "
                     "person. Is that true of the state?",
            options=options, label=0 if has else 1,
            source="pii-masking", task="noul"))
    return out


def contract_clauses(rng, limit):
    """LEDGAR: 100 contract clause types. JevBench 'policy' lives here."""
    from datasets import load_dataset

    data = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
    names = data.features["label"].names
    out = []
    for row in data.select(range(min(limit, len(data)))):
        gold = names[row["label"]]
        others = rng.sample([n for n in names if n != gold], k=min(5, len(names) - 1))
        criteria = {n.replace("_", " "): "" for n in [gold, *others]}
        options = options_from(criteria, rng)
        out.append(Decision(
            id=rid("ledgar", row["text"][:120]), state=row["text"],
            question=pick(rng, "Which kind of contract clause is this?"),
            options=options,
            label=[o.id for o in options].index(gold.replace("_", " ")),
            source="ledgar-clause", task="choice"))
    return out


def answer_adequacy(rng, limit):
    """WikiQA: does this answer the question? JevBench 'adequacy', 7/12 wrong."""
    from datasets import load_dataset

    out = []
    for row in load_dataset("microsoft/wiki_qa", split="train").select(
            range(min(limit, 20360))):
        state = f"Question: {row['question']}\nCandidate answer: {row['answer']}"
        out.append(Decision(
            id=rid("wikiqa", state[:120]), state=state,
            question="The candidate answer actually answers the question. "
                     "Is that true of the state?",
            options=[Option("Yes", "Yes"), Option("No", "No")],
            label=0 if row["label"] == 1 else 1,
            source="wikiqa-adequacy", task="noul"))
    return out


def helpfulness(rng, limit):
    """HelpSteer2: five ordered 0-4 ratings per response. Ordered scales."""
    from datasets import load_dataset

    described = {
        "helpfulness": ["not helpful at all", "barely helpful",
                        "partly helpful", "mostly helpful", "fully helpful"],
        "correctness": ["wrong throughout", "mostly wrong", "partly correct",
                        "mostly correct", "correct throughout"],
        "coherence": ["incoherent", "hard to follow", "understandable",
                      "clear", "perfectly clear"],
        "complexity": ["a child could write it", "simple language",
                       "some expertise", "domain expertise",
                       "deep expertise required"],
        "verbosity": ["far too short", "somewhat short", "about right",
                      "somewhat long", "far too long"],
    }
    out = []
    data = load_dataset("nvidia/HelpSteer2", split="train")
    for row in data.select(range(min(limit, len(data)))):
        aspect = rng.choice(list(described))
        level = row.get(aspect)
        if level is None:
            continue
        words = described[aspect]
        options = [Option(str(i), f"{i}: {w}") for i, w in enumerate(words)]
        state = f"Prompt: {row['prompt']}\n\nResponse: {row['response']}"
        out.append(Decision(
            id=rid("helpsteer", state[:120], aspect), state=state,
            question=f"Rate the {aspect} of the response. "
                     f"The scale runs from {words[0]} to {words[-1]}.",
            options=options, label=int(level), source="helpsteer2", task="score"))
    return out


def fact_check(rng, limit):
    """FEVER with its evidence: supported, refuted, or not enough information."""
    from datasets import load_dataset

    criteria = {"supported": "The evidence establishes the claim",
                "refuted": "The evidence establishes the opposite",
                "not enough info": "The evidence does not settle the claim"}
    mapping = {"SUPPORTS": "supported", "REFUTES": "refuted",
               "NOT ENOUGH INFO": "not enough info"}
    out = []
    data = load_dataset("copenlu/fever_gold_evidence", split="train")
    for row in data.select(range(min(limit, len(data)))):
        gold = mapping.get(row["label"])
        if gold is None:
            continue
        evidence = row.get("evidence") or []
        text = " ".join(str(e[-1]) if isinstance(e, (list, tuple)) else str(e)
                        for e in evidence)[:4000]
        options = options_from(criteria, rng)
        out.append(Decision(
            id=rid("fever", row["claim"][:120]),
            state=f"Evidence: {text}\n\nClaim: {row['claim']}",
            question=pick(rng, "Does the evidence support the claim?"),
            options=options, label=[o.id for o in options].index(gold),
            source="fever-evidence", task="choice"))
    return out


def deception(rng, limit):
    """LIAR2: six-level truthfulness of a public statement. An ordered scale."""
    from datasets import load_dataset

    words = ["pants on fire", "false", "barely true",
             "half true", "mostly true", "true"]
    out = []
    data = load_dataset("chengxuphd/liar2", split="train")
    for row in data.select(range(min(limit, len(data)))):
        level = int(row["label"])
        if not 0 <= level < len(words):
            continue
        state = row["statement"]
        if row.get("speaker"):
            state = f"{row['speaker']} said: {state}"
        options = [Option(str(i), f"{i}: {w}") for i, w in enumerate(words)]
        out.append(Decision(
            id=rid("liar2", state[:120]), state=state,
            question="How truthful is this statement? The scale runs from "
                     "pants on fire to true.",
            options=options, label=level, source="liar2", task="score"))
    return out


def intents_with_abstain(rng, limit):
    """CLINC-150 plus its out-of-scope split: when nothing on offer fits."""
    from datasets import load_dataset

    data = load_dataset("clinc/clinc_oos", "plus", split="train")
    names = data.features["intent"].names
    oos = [i for i, n in enumerate(names) if n == "oos"]
    out = []
    for row in data.select(range(min(limit, len(data)))):
        gold = names[row["intent"]]
        pool = [n for n in names if n not in ("oos", gold)]
        criteria = {n.replace("_", " "): "" for n in rng.sample(pool, 5)}
        criteria["none of these"] = "The request does not match any option"
        if gold != "oos":
            criteria[gold.replace("_", " ")] = ""
        options = options_from(criteria, rng)
        target = "none of these" if gold == "oos" else gold.replace("_", " ")
        out.append(Decision(
            id=rid("clinc", row["text"]), state=row["text"],
            question=pick(rng, "What is the user asking for?"),
            options=options, label=[o.id for o in options].index(target),
            source="clinc-oos" if gold == "oos" else "clinc-intent",
            task="choice"))
    _ = oos
    return out


def toxicity_facets(rng, limit):
    """Civil comments: several ordered harm ratings on the same text."""
    from datasets import load_dataset

    facets = ["toxicity", "severe_toxicity", "obscene", "threat", "insult",
              "identity_attack"]
    words = ["none of it", "a little", "clearly", "severely"]
    out = []
    stream = load_dataset("google/civil_comments", split="train", streaming=True)
    for row in stream:
        if len(out) >= limit:
            break
        facet = rng.choice(facets)
        value = row.get(facet)
        if value is None or not row["text"].strip():
            continue
        level = min(3, int(float(value) * 4)) if float(value) < 1.0 else 3
        options = [Option(str(i), f"{i}: {w}") for i, w in enumerate(words)]
        out.append(Decision(
            id=rid("civil", row["text"][:120], facet), state=row["text"],
            question=f"How much {facet.replace('_', ' ')} does this comment "
                     f"contain? The scale runs from none of it to severely.",
            options=options, label=level, source="civil-comments", task="score"))
    return out


def emotions(rng, limit):
    """GoEmotions: 28 fine emotions, a large label set with real meaning."""
    from datasets import load_dataset

    data = load_dataset("google-research-datasets/go_emotions", "simplified",
                        split="train")
    names = data.features["labels"].feature.names
    out = []
    for row in data.select(range(min(limit, len(data)))):
        if len(row["labels"]) != 1:
            continue
        gold = names[row["labels"][0]]
        others = rng.sample([n for n in names if n != gold], 5)
        criteria = {n: "" for n in [gold, *others]}
        options = options_from(criteria, rng)
        out.append(Decision(
            id=rid("goemotions", row["text"][:120]), state=row["text"],
            question=pick(rng, "Which emotion does this text express?"),
            options=options, label=[o.id for o in options].index(gold),
            source="go-emotions", task="choice"))
    return out


def topics(rng, limit):
    """Yahoo answers: ten topics over a question and its best answer."""
    from datasets import load_dataset

    names = ["society and culture", "science and mathematics", "health",
             "education and reference", "computers and internet", "sports",
             "business and finance", "entertainment and music",
             "family and relationships", "politics and government"]
    out = []
    stream = load_dataset("community-datasets/yahoo_answers_topics",
                          split="train", streaming=True)
    for row in stream:
        if len(out) >= limit:
            break
        gold = names[int(row["topic"])]
        state = f"{row['question_title']}\n{row['question_content']}".strip()
        if not state:
            continue
        criteria = {n: "" for n in names}
        options = options_from(criteria, rng)
        out.append(Decision(
            id=rid("yahoo", state[:120]), state=state,
            question=pick(rng, "Which topic does this question belong to?"),
            options=options, label=[o.id for o in options].index(gold),
            source="yahoo-topics", task="choice"))
    return out


def stars(rng, limit):
    """Amazon reviews: a five-star ordered scale with named levels."""
    from datasets import load_dataset

    words = ["one star, terrible", "two stars, poor", "three stars, mixed",
             "four stars, good", "five stars, excellent"]
    out = []
    data = load_dataset("SetFit/amazon_reviews_multi_en", split="train")
    for row in data.select(range(min(limit, len(data)))):
        level = int(row["label"])
        options = [Option(str(i + 1), f"{i + 1}: {w}") for i, w in enumerate(words)]
        out.append(Decision(
            id=rid("amazon", row["text"][:120]), state=row["text"],
            question="How many stars does this review give? The scale runs "
                     "from one star to five stars.",
            options=options, label=level, source="amazon-stars", task="score"))
    return out


def finance(rng, limit):
    """Financial news sentiment: a small domain nothing else in the mix has."""
    from datasets import load_dataset

    criteria = {"bearish": "Bad news for the price",
                "bullish": "Good news for the price",
                "neutral": "Neither good nor bad for the price"}
    names = ["bearish", "bullish", "neutral"]
    out = []
    data = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    for row in data.select(range(min(limit, len(data)))):
        options = options_from(criteria, rng)
        gold = names[int(row["label"])]
        out.append(Decision(
            id=rid("finance", row["text"][:120]), state=row["text"],
            question=pick(rng, "What does this financial headline mean for the price?"),
            options=options, label=[o.id for o in options].index(gold),
            source="finance-sentiment", task="choice"))
    return out


SOURCES = {
    "grammar": grammar, "pii": pii, "contracts": contract_clauses,
    "adequacy": answer_adequacy, "helpfulness": helpfulness,
    "fact_check": fact_check, "deception": deception,
    "intents": intents_with_abstain, "toxicity": toxicity_facets,
    "emotions": emotions, "topics": topics, "stars": stars, "finance": finance,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-train", default="data/domains.jsonl")
    parser.add_argument("--out-test", default="data/test_domains.jsonl")
    parser.add_argument("--per-source", type=int, default=12000)
    parser.add_argument("--test-per-source", type=int, default=250)
    parser.add_argument("--only", default="", help="comma-separated source names")
    parser.add_argument("--seed", type=int, default=523)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    wanted = [s.strip() for s in args.only.split(",") if s.strip()] or list(SOURCES)
    train, test = [], []
    for name in wanted:
        builder = SOURCES.get(name)
        if builder is None:
            print(f"  {name}: unknown source")
            continue
        try:
            rows = builder(rng, args.per_source + args.test_per_source)
        except Exception as error:
            print(f"  {name}: failed ({str(error).splitlines()[0][:90]})", flush=True)
            continue
        rng.shuffle(rows)
        test += rows[: args.test_per_source]
        train += rows[args.test_per_source :]
        print(f"  {name}: {len(rows)} rows", flush=True)

    rng.shuffle(train)
    rng.shuffle(test)
    write_jsonl(args.out_train, train)
    write_jsonl(args.out_test, test)
    print(f"\nwrote {len(train)} training rows and {len(test)} test rows")


if __name__ == "__main__":
    main()
