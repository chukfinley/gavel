"""Loss and calibration metrics.

Cross entropy alone gives a model that picks the right option but says
nothing honest about how sure it is. The Brier score is a proper scoring rule:
it is smallest when the reported probability equals the true probability.
Training on a mix of both gives an argmax that is still sharp and a
probability that can be used as a threshold.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def brier(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=-1) * mask
    target = F.one_hot(labels, num_classes=logits.size(-1)).float()
    return ((probabilities - target) ** 2 * mask).sum(dim=-1).mean()


class DecisionLoss(nn.Module):
    def __init__(self, brier_weight: float = 0.5, label_smoothing: float = 0.0):
        super().__init__()
        self.brier_weight = brier_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels, mask) -> tuple[torch.Tensor, dict]:
        ce = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        br = brier(logits, labels, mask)
        total = ce + self.brier_weight * br
        return total, {"ce": ce.item(), "brier": br.item()}


def expected_calibration_error(probabilities, labels, bins: int = 15) -> float:
    """Mean gap between confidence and accuracy, over equal-width bins."""
    confidence, prediction = probabilities.max(dim=-1)
    correct = (prediction == labels).float()
    error, total = 0.0, confidence.numel()
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = (confidence > low) & (confidence <= high)
        if selected.any():
            gap = (confidence[selected].mean() - correct[selected].mean()).abs()
            error += gap.item() * selected.sum().item() / total
    return error


def balanced_accuracy(prediction: torch.Tensor, labels: torch.Tensor) -> float:
    scores = []
    for value in labels.unique():
        selected = labels == value
        scores.append((prediction[selected] == value).float().mean().item())
    return sum(scores) / len(scores) if scores else 0.0
