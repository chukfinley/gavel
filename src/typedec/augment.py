"""Question augmentation, so that the model reads the question.

A model trained on a fixed set of phrasings learns to recognise the question
instead of reading it, and then fails the moment a caller words it differently.
The open-jev report measures exactly this gap: 0.854 in domain against 0.690 on
unseen instructions and option sets.

Every transform here keeps the gold answer correct by construction:

* the question is paraphrased, the answer does not move,
* wrong options are dropped, the correct one stays,
* a yes-no question is negated and the label flips with it,
* option wording is replaced by a synonym of the same option.
"""

from __future__ import annotations

import random

from .schema import Decision, Option

PARAPHRASES = [
    "{q}",
    "Question: {q}",
    "Decide: {q}",
    "{q} Pick exactly one.",
    "Answer this about the text above: {q}",
    "Based on the information given, {q_lower}",
]

YES_WORDS = {"yes", "true", "correct"}
NO_WORDS = {"no", "false", "incorrect"}


def paraphrase(decision: Decision, rng: random.Random) -> Decision:
    question = decision.question.strip()
    template = rng.choice(PARAPHRASES)
    lowered = question[0].lower() + question[1:] if question else question
    decision.question = template.format(q=question, q_lower=lowered)
    return decision


def drop_distractors(decision: Decision, rng: random.Random) -> Decision:
    """Remove wrong options. The correct one is never removed."""
    if len(decision.options) < 3 or decision.label is None:
        return decision
    keep = max(2, rng.randint(2, len(decision.options) - 1))
    wrong = [i for i in range(len(decision.options)) if i != decision.label]
    rng.shuffle(wrong)
    chosen = sorted(wrong[: keep - 1] + [decision.label])
    decision.options = [decision.options[i] for i in chosen]
    decision.label = chosen.index(decision.label)
    return decision


def negate(decision: Decision, rng: random.Random) -> Decision:
    """Turn a yes-no question around and move the label with it."""
    ids = {option.id.lower() for option in decision.options}
    if len(decision.options) != 2 or not (ids & YES_WORDS and ids & NO_WORDS):
        return decision
    decision.question = f"It is not the case that: {decision.question.rstrip('?')}?"
    decision.label = 1 - decision.label
    return decision


SYNONYMS = {
    "Yes": ["Yes", "True", "That is correct"],
    "No": ["No", "False", "That is not correct"],
}


def reword_options(decision: Decision, rng: random.Random) -> Decision:
    decision.options = [
        Option(option.id, rng.choice(SYNONYMS.get(option.description, [option.description])))
        for option in decision.options
    ]
    return decision


TRANSFORMS = [paraphrase, drop_distractors, negate, reword_options]


def augment(decision: Decision, rng: random.Random, probability: float = 0.7) -> Decision:
    """Apply one or two transforms, keeping the gold answer correct."""
    if rng.random() > probability:
        return decision
    copy = Decision(id=decision.id, state=decision.state, question=decision.question,
                    options=list(decision.options), label=decision.label,
                    source=decision.source, task=decision.task, meta=dict(decision.meta))
    for transform in rng.sample(TRANSFORMS, rng.choice([1, 1, 2])):
        copy = transform(copy, rng)
    return copy
