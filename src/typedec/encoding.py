"""Turn decisions into model batches."""

from __future__ import annotations

import random

import torch

from .model import OPT_TOKEN, Batch
from .schema import Decision


def render(decision: Decision, order: list[int]) -> tuple[str, list[str]]:
    """Return the text before the options, and one text for each option.

    The marker itself is not part of these strings. It is inserted as a token
    id during encoding, because a tokenizer may merge a marker with the space
    in front of it and then the recorded position would be wrong.
    """
    head = f"Question: {decision.question}\nSituation: {decision.state}\nOptions:"
    parts = [f" {decision.options[i].description}" for i in order]
    return head, parts


def encode(
    decisions: list[Decision],
    tokenizer,
    max_length: int = 512,
    shuffle_options: bool = False,
    rng: random.Random | None = None,
) -> Batch:
    rng = rng or random.Random(0)
    opt_id = tokenizer.convert_tokens_to_ids(OPT_TOKEN)
    max_options = max(len(d.options) for d in decisions)

    all_ids, all_positions, all_masks, labels = [], [], [], []
    for decision in decisions:
        order = list(range(len(decision.options)))
        if shuffle_options:
            rng.shuffle(order)
        head, parts = render(decision, order)

        head_ids = tokenizer.encode(head, add_special_tokens=False)
        option_ids = [[opt_id] + tokenizer.encode(p, add_special_tokens=False) for p in parts]
        budget = max_length - 2 - sum(len(o) for o in option_ids)
        if budget < 16:
            # Options alone fill the window: cut each option text instead.
            keep = max(8, (max_length - 2 - len(head_ids)) // max(len(option_ids), 1))
            option_ids = [o[:keep] for o in option_ids]   # the marker is index 0, thus kept
            budget = max_length - 2 - sum(len(o) for o in option_ids)
        head_ids = head_ids[:max(budget, 0)]

        ids = [tokenizer.cls_token_id] + head_ids
        positions = []
        for chunk in option_ids:
            positions.append(len(ids))          # the [OPT] marker starts each chunk
            ids.extend(chunk)
        ids.append(tokenizer.sep_token_id)
        assert all(ids[p] == opt_id for p in positions), "marker position drifted"

        if decision.label is not None:
            labels.append(order.index(decision.label))
        all_ids.append(ids)
        all_positions.append(positions)
        all_masks.append([1] * len(positions))

    width = max(len(i) for i in all_ids)
    pad = tokenizer.pad_token_id
    input_ids = torch.full((len(all_ids), width), pad, dtype=torch.long)
    attention = torch.zeros((len(all_ids), width), dtype=torch.long)
    positions = torch.full((len(all_ids), max_options), -1, dtype=torch.long)
    option_mask = torch.zeros((len(all_ids), max_options), dtype=torch.long)
    for row, (ids, pos, mask) in enumerate(zip(all_ids, all_positions, all_masks)):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention[row, : len(ids)] = 1
        positions[row, : len(pos)] = torch.tensor(pos)
        option_mask[row, : len(mask)] = torch.tensor(mask)

    return Batch(
        input_ids=input_ids,
        attention_mask=attention,
        option_positions=positions,
        option_mask=option_mask,
        labels=torch.tensor(labels, dtype=torch.long) if labels else None,
    )
