"""The outside teacher's distribution follows the options through augmentation."""
import random

import torch

from gavel import teacher as teaching
from gavel.augment import drop_distractors, negate
from gavel.schema import Decision, Option


def row(n=4, label=2):
    return Decision(id="r", state="s", question="q?",
                    options=[Option(f"o{i}", f"option {i}") for i in range(n)], label=label,
                    meta={"teacher": [0.1, 0.2, 0.6, 0.1]})


def test_drop_distractors_keeps_the_teacher_aligned():
    d = drop_distractors(row(), random.Random(1))
    kept = [o.description for o in d.options]
    assert "option 2" in kept and len(d.meta["teacher"]) == len(kept)
    assert abs(sum(d.meta["teacher"]) - 1.0) < 1e-6
    assert d.meta["teacher"][d.label] == max(d.meta["teacher"])


def test_negate_swaps_the_two_probabilities():
    d = Decision(id="r", state="s", question="Is it?", label=0,
                 options=[Option("Yes", "Yes"), Option("No", "No")], meta={"teacher": [0.8, 0.2]})
    d = negate(d, random.Random(0))
    assert d.label == 1 and d.meta["teacher"] == [0.2, 0.8]


def test_soft_targets_follow_a_permutation_and_kl_is_zero_at_the_teacher():
    soft, has = teaching.soft_targets([row()], 5, permutations=[[3, 2, 1, 0]])
    assert has.tolist() == [True]
    assert torch.allclose(soft[0, :4], torch.tensor([0.1, 0.6, 0.2, 0.1])) and soft[0, 4] == 0
    valid = torch.tensor([[1, 1, 1, 1, 0]])
    logp = soft.clamp_min(1e-8).log().masked_fill(valid == 0, float("-inf"))
    assert teaching.kl_to_teacher(logp, soft, valid, has).item() < 1e-5
    # A row without a teacher contributes nothing and does not break the mean.
    plain = row(); plain.meta = {}
    soft2, has2 = teaching.soft_targets([plain], 5)
    assert not has2.any()
    assert teaching.kl_to_teacher(logp, soft2, valid, has2).item() == 0.0
