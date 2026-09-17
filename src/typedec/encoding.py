"""Build encoder batches for decisions and for the entailment anchor."""

from __future__ import annotations

import torch

from .schema import Decision

NLI_SOURCES = {"mnli", "wanli-train", "anli-r3", "snli"}
CLAIM_PREFIX = "Assess the claim: "


def hypothesis(decision: Decision, index: int) -> str:
    """The sentence whose entailment against the state scores this option.

    Rows that came from inference data already carry a full statement as the
    option text; they read well without a template. Everything else is joined
    with the question, because the option alone ("Billing support") is not a
    statement about the state.
    """
    text = decision.options[index].description.strip()
    question = decision.question.strip()
    if question.startswith(CLAIM_PREFIX):
        return f"{question[len(CLAIM_PREFIX):].rstrip('.')}. {text}."
    return f"{question} The answer is {text}."


def premise(decision: Decision) -> str:
    return decision.state.strip() or decision.question.strip()


def encode_options(decisions: list[Decision], tokenizer, max_length: int = 256):
    """One (premise, hypothesis) pair for every option of every decision."""
    width = max(len(d.options) for d in decisions)
    premises, hypotheses, mask = [], [], []
    for decision in decisions:
        state = premise(decision)
        for index in range(width):
            premises.append(state)
            hypotheses.append(hypothesis(decision, index) if index < len(decision.options) else "")
        mask.append([1] * len(decision.options) + [0] * (width - len(decision.options)))
    encoding = tokenizer(premises, hypotheses, truncation=True, max_length=max_length,
                         padding=True, return_tensors="pt")
    labels = torch.tensor([d.label for d in decisions]) if decisions[0].label is not None else None
    return encoding, torch.tensor(mask), labels


def encode_anchor(decisions: list[Decision], tokenizer, max_length: int = 256):
    """Natural language inference pairs, used to anchor the three-way head."""
    premises = [d.state for d in decisions]
    hypotheses = [d.question.replace(CLAIM_PREFIX, "").strip() for d in decisions]
    encoding = tokenizer(premises, hypotheses, truncation=True, max_length=max_length,
                         padding=True, return_tensors="pt")
    return encoding, torch.tensor([d.label for d in decisions])


def to_device(encoding, device):
    return {key: value.to(device) for key, value in encoding.items()}
