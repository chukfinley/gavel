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


def option_pairs(decisions: list[Decision]):
    """One (premise, hypothesis) pair per REAL option, and the layout mask.

    The first version padded every decision to the widest one in the batch
    with empty hypotheses that ran through the encoder and were then masked
    to minus infinity: 28 percent of the sequences in a typical batch, zero
    gradient. Only real pairs are built now, in row-major order, and the
    mask says where each one goes.
    """
    width = max(len(d.options) for d in decisions)
    premises, hypotheses, mask = [], [], []
    for decision in decisions:
        state = premise(decision)
        for index in range(len(decision.options)):
            premises.append(state)
            hypotheses.append(hypothesis(decision, index))
        mask.append([1] * len(decision.options) + [0] * (width - len(decision.options)))
    return premises, hypotheses, torch.tensor(mask)


def encode_options(decisions: list[Decision], tokenizer, max_length: int = 256):
    """Tokenized pairs for every real option, plus the (rows, width) mask."""
    premises, hypotheses, mask = option_pairs(decisions)
    encoding = tokenizer(premises, hypotheses, truncation=True, max_length=max_length,
                         padding=True, return_tensors="pt")
    labels = torch.tensor([d.label for d in decisions]) if decisions[0].label is not None else None
    return encoding, mask, labels


def anchor_pairs(decisions: list[Decision]):
    premises = [d.state for d in decisions]
    hypotheses = [d.question.replace(CLAIM_PREFIX, "").strip() for d in decisions]
    return premises, hypotheses, torch.tensor([d.label for d in decisions])


def encode_anchor(decisions: list[Decision], tokenizer, max_length: int = 256):
    """Natural language inference pairs, used to anchor the three-way head."""
    premises, hypotheses, labels = anchor_pairs(decisions)
    encoding = tokenizer(premises, hypotheses, truncation=True, max_length=max_length,
                         padding=True, return_tensors="pt")
    return encoding, labels


def to_device(encoding, device):
    """Pinned, non-blocking copies: a plain .to() from pageable memory is a
    synchronous call that drains the GPU queue every step."""
    cuda = str(device).startswith("cuda")
    return {key: (value.pin_memory().to(device, non_blocking=True) if cuda
                  else value.to(device))
            for key, value in encoding.items()}
