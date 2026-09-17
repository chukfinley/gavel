"""Typed decision model built on an entailment scorer.

Design note, and the reason for the rewrite: a head that reads option markers
inside one sequence does not train. The first version put the option texts into
one sequence and scored each marker. Cross entropy stayed at ln(number of
options) for thousands of steps — the model gives a uniform answer. The same
happens with a span-pooled head and with the standard multiple-choice head of
`transformers`. The cause is the option text: a string like "The evidence
establishes the claim" means nothing to an untrained head, thus every option
keeps the same score and there is no gradient that separates them.

Zero-shot classifiers solve this the other way round. The option becomes a
*hypothesis about the state*, and a three-way entailment head scores it. The
head has a fixed meaning (entailment, neutral, contradiction), which natural
language inference data anchors, and the option text lives where the model
already understands text. The score of an option is its entailment logit; a
softmax over the options gives the decision.

Two properties follow:

* The answer space is defined at run time, because the option text is part of
  the input.
* The result cannot depend on the order of the options, because each option is
  scored in its own sequence. Position bias is impossible by construction.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ENTAILMENT, NEUTRAL, CONTRADICTION = 0, 1, 2


class EntailmentScorer(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()
        self.backbone_name = backbone
        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=3, trust_remote_code=True)
        self.register_buffer("log_temperature", torch.zeros(1))

    def pair_logits(self, encoding: dict) -> torch.Tensor:
        """Three-way logits for a batch of (premise, hypothesis) pairs."""
        return self.model(**encoding).logits

    def option_logits(self, encoding: dict, option_mask: torch.Tensor) -> torch.Tensor:
        """One score for each option: the entailment logit of its hypothesis."""
        rows, options = option_mask.shape
        scores = self.pair_logits(encoding)[:, ENTAILMENT].view(rows, options)
        scores = scores / self.log_temperature.exp()
        return scores.masked_fill(option_mask == 0, torch.finfo(scores.dtype).min)


def build_tokenizer(backbone: str):
    return AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
