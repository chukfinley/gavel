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

import importlib.util
import os

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ENTAILMENT, NEUTRAL, CONTRADICTION = 0, 1, 2


def attention_implementation() -> str | None:
    """Flash-attention 2 when it is installed, otherwise the library default.

    With the default (sdpa) every batch that carries a padding mask or a
    sliding-window mask — which is every batch here — falls back to the
    memory-efficient kernel, which skips nothing: the fourteen local layers
    then compute full attention over 8192 tokens. Flash-attention gets the
    64-token window natively and unpads per layer. `GAVEL_ATTN` overrides.
    """
    forced = os.environ.get("GAVEL_ATTN")
    if forced:
        return None if forced == "default" else forced
    if not torch.cuda.is_available():
        return None                        # the kernel needs a card; CPU evals stay on sdpa
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else None


def set_layer_checkpointing(module: nn.Module, on: bool) -> int:
    """Flip gradient checkpointing per layer without re-registering hooks.

    `gradient_checkpointing_enable()` registers input-grad hooks every time
    it is called; toggled per step they would stack up over a run. The
    layers check a plain attribute, so that is what is flipped.
    """
    count = 0
    for child in module.modules():
        if child is not module and hasattr(child, "gradient_checkpointing") \
                and not hasattr(child, "config"):
            child.gradient_checkpointing = on
            count += 1
    return count


class EntailmentScorer(nn.Module):
    """Works with an encoder backbone and with a decoder backbone.

    A decoder (Qwen and friends) has no padding token and pools the last
    token instead of the first, which `AutoModelForSequenceClassification`
    handles once the padding id is set.
    """

    def __init__(self, backbone: str, gradient_checkpointing: bool = False):
        super().__init__()
        self.backbone_name = backbone
        extra = {"attn_implementation": attention_implementation()}
        extra = {k: v for k, v in extra.items() if v}
        self.model = AutoModelForSequenceClassification.from_pretrained(
            backbone, num_labels=3, trust_remote_code=True, **extra)
        # Some multimodal configurations keep the text settings in a sub-config
        # and have no pad id at the top level.
        for config in filter(None, [self.model.config,
                                    getattr(self.model.config, "text_config", None)]):
            if getattr(config, "pad_token_id", None) is None:
                config.pad_token_id = getattr(config, "eos_token_id", None) or 0
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        self.register_buffer("log_temperature", torch.zeros(1))

    def set_checkpointing(self, on: bool) -> None:
        """Checkpointing only where the memory is needed (the long steps)."""
        set_layer_checkpointing(self.model, on)

    def pair_logits(self, encoding: dict) -> torch.Tensor:
        """Three-way logits for a batch of (premise, hypothesis) pairs."""
        return self.model(**encoding).logits

    def option_logits(self, encoding: dict, option_mask: torch.Tensor) -> torch.Tensor:
        """One score for each option: the entailment logit of its hypothesis.

        The encoding holds only the real pairs, in row-major order of the
        mask; the mask says where each one lands. Empty slots stay at the
        lowest value, as before, but no longer cost a forward pass.
        """
        flat = self.pair_logits(encoding)[:, ENTAILMENT]
        return self.scatter_options(flat, option_mask)

    def scatter_options(self, flat: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        """Place each real pair's score at its (row, option) slot, in fp32.

        Always fp32: under bf16 autocast the logits arrive in bf16 and the
        temperature is an fp32 buffer, and an index assignment between the
        two is an error, which is how the first optimised run died at its
        first evaluation.
        """
        flat = flat.float()
        scores = torch.full(option_mask.shape, torch.finfo(torch.float32).min,
                            dtype=torch.float32, device=flat.device)
        scores[option_mask.to(flat.device).bool()] = (
            flat / self.log_temperature.exp().to(flat.device))
        return scores

def build_tokenizer(backbone: str):
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
