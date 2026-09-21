"""Soft labels from an outside teacher, kept next to each row.

`scripts/label_with_jev.py` writes one line per row id with the teacher's
probability per option, in our option order. `attach()` puts that list
into `row.meta["teacher"]`, where the augmentations keep it aligned with
the options, and `soft_targets()` turns a batch back into a tensor.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def load(path: str | Path) -> dict[str, list[float]]:
    table: dict[str, list[float]] = {}
    for line in open(path):
        try:
            record = json.loads(line)
        except Exception:
            continue
        probabilities = record.get("teacher", {}).get("probabilities")
        if probabilities:
            table[record["id"]] = [float(p) for p in probabilities]
    return table


def attach(rows, table: dict[str, list[float]]) -> list:
    """Rows that carry a teacher distribution, after attaching it."""
    labelled = []
    for row in rows:
        probabilities = table.get(row.id)
        if probabilities and len(probabilities) == len(row.options):
            row.meta["teacher"] = probabilities
            labelled.append(row)
    return labelled


def soft_targets(rows, width: int, permutations=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(batch, width) probabilities and a (batch,) mask of rows that have them.

    `permutations` gives, per row, the option index that landed in each
    slot, for trainers that shuffle option order.
    """
    soft = torch.zeros(len(rows), width)
    has = torch.zeros(len(rows), dtype=torch.bool)
    for i, row in enumerate(rows):
        teacher = row.meta.get("teacher")
        if not teacher or len(teacher) != len(row.options):
            continue
        order = permutations[i] if permutations is not None else range(len(teacher))
        values = [teacher[j] for j in order]
        soft[i, : len(values)] = torch.tensor(values)
        has[i] = True
    return soft, has


def kl_to_teacher(log_probabilities: torch.Tensor, soft: torch.Tensor,
                  valid: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) over real option slots, mean over teacher rows."""
    if not has.any():
        # Not `log_probabilities.sum() * 0`: padded slots are -inf and
        # -inf * 0 is NaN, which would poison the whole step.
        return torch.zeros((), device=log_probabilities.device)
    valid = valid.float()
    safe_log = log_probabilities.masked_fill(valid == 0, 0.0)
    soft = soft * valid
    kl = (soft * (soft.clamp_min(1e-8).log() - safe_log) * valid).sum(-1)
    return (kl * has.float()).sum() / has.float().sum()
