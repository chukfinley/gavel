"""Typed decision model: one forward pass, one score for each option.

The input holds the question, the state, and then one segment for each option.
Every option segment starts with the marker token `[OPT]`. The hidden state at
each marker goes through one shared scoring head, which gives one logit for
each option. A softmax over these logits gives the probabilities.

Because the option text is part of the input, the answer space can change with
every request. The model never learns a fixed set of classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

OPT_TOKEN = "[OPT]"


@dataclass
class Batch:
    input_ids: torch.Tensor        # (B, L)
    attention_mask: torch.Tensor   # (B, L)
    option_positions: torch.Tensor # (B, K) index of each [OPT] marker, -1 when padded
    option_mask: torch.Tensor      # (B, K) 1 for a real option
    labels: torch.Tensor | None = None

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            self.input_ids.to(device),
            self.attention_mask.to(device),
            self.option_positions.to(device),
            self.option_mask.to(device),
            None if self.labels is None else self.labels.to(device),
        )


class OptionScorer(nn.Module):
    def __init__(self, backbone: str, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(backbone, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(backbone, trust_remote_code=True)
        hidden = getattr(self.config, "hidden_size", None) or self.config.d_model
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # One learned temperature. Calibration on a held-out split only scales
        # this value; the weights stay untouched.
        self.log_temperature = nn.Parameter(torch.zeros(1), requires_grad=False)

    def resize(self, tokenizer) -> None:
        self.backbone.resize_token_embeddings(len(tokenizer))

    def forward(self, batch: Batch) -> torch.Tensor:
        out = self.backbone(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        states = out.last_hidden_state                       # (B, L, H)
        positions = batch.option_positions.clamp(min=0)      # (B, K)
        gathered = torch.gather(
            states, 1, positions.unsqueeze(-1).expand(-1, -1, states.size(-1))
        )                                                    # (B, K, H)
        logits = self.head(gathered).squeeze(-1)             # (B, K)
        logits = logits / self.log_temperature.exp()
        return logits.masked_fill(batch.option_mask == 0, torch.finfo(logits.dtype).min)


def build_tokenizer(backbone: str):
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if OPT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [OPT_TOKEN]})
    return tokenizer
