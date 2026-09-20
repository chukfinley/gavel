"""One sequence, every option, one forward pass.

This replaces the pair-per-option scorer for the short-state path. The state
is read once and every option is scored from the same hidden states:

    [CLS] [STATE] state [Q] question [OPT] option [OPT] option … [SEP]

An option's score comes from three things pooled over token spans: the mean
of its own tokens, the mean of its question's tokens, and their elementwise
product. A softmax inside each question's group of options is the answer.

## Why this trains where the first attempt did not

`model.py` records the failure: a head that reads an option *marker token*
keeps every option at the same score and cross entropy sits at ln(number of
options) forever. The reason is that an untrained head sees no relation
between the question and the option — a marker embedding carries no
interaction, so there is no gradient that separates one option from another.

The product term is that interaction, the same trick a sentence-pair
bi-encoder uses with `[u; v; u*v]`. It is what kotoba's working span head
does, and the difference between a marker head and a span head is exactly
this, not the idea of one sequence.

Two properties change and both need watching:

* **Order can now matter.** Every option sits at a different position in the
  same sequence, so position bias is possible, where the pair scorer made it
  impossible by construction. Training shuffles the option order, and
  `flip_rate` in the evaluation measures what is left.
* **Options share a window.** The state plus every option must fit in the
  context, so a very long state with many long options truncates the state.
  The pair scorer gave each option the whole window.

What is bought: the state is encoded once instead of once per option, and
several questions about the same state ride in the same sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

STATE, QUESTION, OPTION = "[STATE]", "[Q]", "[OPT]"
MARKERS = [STATE, QUESTION, OPTION]


@dataclass
class Encoded:
    input_ids: torch.Tensor          # (batch, length)
    attention_mask: torch.Tensor     # (batch, length)
    option_mask: torch.Tensor        # (batch, options, length) 1 on option tokens
    question_mask: torch.Tensor      # (batch, options, length) 1 on its question
    group: torch.Tensor              # (batch, options) question index, -1 = padding
    counts: list[list[int]]          # options per question, per row

    def to(self, device) -> "Encoded":
        return Encoded(self.input_ids.to(device), self.attention_mask.to(device),
                       self.option_mask.to(device), self.question_mask.to(device),
                       self.group.to(device), self.counts)


def build_tokenizer(backbone: str):
    tokenizer = AutoTokenizer.from_pretrained(backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})
    if added:
        pass                              # the model resizes its embeddings
    return tokenizer


def encode(tokenizer, rows, max_length: int = 1024, device=None) -> Encoded:
    """Build one sequence per row.

    `rows` is a list of `(state, [(question, [option, ...]), ...])`. The state
    is truncated, never the options: an option that is cut cannot be scored,
    while a state that is cut only loses evidence.
    """
    marker_ids = {name: tokenizer.convert_tokens_to_ids(name) for name in MARKERS}
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    sequences, spans_per_row = [], []
    for state, questions in rows:
        tail: list[int] = []
        spans: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        for index, (question, options) in enumerate(questions):
            question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
            start = len(tail) + 1                       # after the [Q] marker
            tail += [marker_ids[QUESTION]] + question_ids
            question_span = (start, len(tail))
            for option in options:
                option_ids = tokenizer(option, add_special_tokens=False)["input_ids"]
                begin = len(tail) + 1                   # after the [OPT] marker
                tail += [marker_ids[OPTION]] + option_ids
                spans.append(((begin, len(tail)), question_span, index))
        # Everything above is fixed; the state gets what is left.
        room = max_length - len(tail) - 3               # [CLS] [STATE] … [SEP]
        state_ids = tokenizer(state or "", add_special_tokens=False,
                              truncation=True, max_length=max(room, 8))["input_ids"]
        head = ([cls_id] if cls_id is not None else []) + [marker_ids[STATE]] + state_ids
        shift = len(head)
        body = head + tail + ([sep_id] if sep_id is not None else [])
        sequences.append(body)
        spans_per_row.append([((a + shift, b + shift), (c + shift, d + shift), g)
                              for (a, b), (c, d), g in spans])

    length = max(len(s) for s in sequences)
    width = max(len(s) for s in spans_per_row)
    batch = len(sequences)
    input_ids = torch.full((batch, length), pad_id, dtype=torch.long)
    attention = torch.zeros((batch, length), dtype=torch.long)
    option_mask = torch.zeros((batch, width, length))
    question_mask = torch.zeros((batch, width, length))
    group = torch.full((batch, width), -1, dtype=torch.long)
    counts = []
    for row, (body, spans) in enumerate(zip(sequences, spans_per_row)):
        input_ids[row, : len(body)] = torch.tensor(body)
        attention[row, : len(body)] = 1
        per_question: dict[int, int] = {}
        for slot, ((a, b), (c, d), index) in enumerate(spans):
            option_mask[row, slot, a:b] = 1
            question_mask[row, slot, c:d] = 1
            group[row, slot] = index
            per_question[index] = per_question.get(index, 0) + 1
        counts.append([per_question[k] for k in sorted(per_question)])

    encoded = Encoded(input_ids, attention, option_mask, question_mask, group, counts)
    return encoded.to(device) if device else encoded


class SpanScorer(nn.Module):
    """An encoder plus a three-layer head over pooled spans."""

    def __init__(self, backbone: str, tokenizer=None, dropout: float = 0.1,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.backbone_name = backbone
        self.model = AutoModel.from_pretrained(backbone, trust_remote_code=True)
        if tokenizer is not None:
            self.model.resize_token_embeddings(len(tokenizer))
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
        hidden = self.model.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1))
        self.register_buffer("log_temperature", torch.zeros(1))

    @staticmethod
    def _pool(mask: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Mean of the hidden states under each mask row."""
        total = torch.bmm(mask, hidden)                        # (b, options, h)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return total / count

    def forward(self, encoded: Encoded) -> torch.Tensor:
        """A score per option slot; padding slots get the lowest value."""
        hidden = self.model(input_ids=encoded.input_ids,
                            attention_mask=encoded.attention_mask).last_hidden_state
        option = self._pool(encoded.option_mask.to(hidden.dtype), hidden)
        question = self._pool(encoded.question_mask.to(hidden.dtype), hidden)
        features = torch.cat([question, option, question * option], dim=-1)
        scores = self.head(features).squeeze(-1).float()
        return scores.masked_fill(encoded.group < 0, torch.finfo(scores.dtype).min)

    def group_log_softmax(self, scores: torch.Tensor,
                          group: torch.Tensor) -> torch.Tensor:
        """Normalise inside each question's options, not across the row."""
        scores = scores / self.log_temperature.exp()
        out = torch.full_like(scores, float("-inf"))
        for index in range(int(group.max().item()) + 1 if group.numel() else 0):
            selected = group == index
            if not selected.any():
                continue
            masked = scores.masked_fill(~selected, torch.finfo(scores.dtype).min)
            out = torch.where(selected, F.log_softmax(masked, dim=-1), out)
        return out
