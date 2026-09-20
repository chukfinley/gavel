"""Tests that run without a download: a tiny random model of the same shape."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from gavel import Gavel
from gavel.schema import Decision, Option, from_json


def test_record_round_trip():
    row = {"id": "x", "state": "s", "question": "q",
           "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}],
           "label": 1}
    decision = from_json(row)
    assert decision.options[decision.label].id == "b"
    assert decision.to_json()["label"] == 1


def test_hypothesis_uses_the_claim_wording():
    judge = Gavel.__new__(Gavel)
    assert judge._hypothesis("Assess the claim: it rained", "Supported") == \
        "it rained. Supported."
    assert judge._hypothesis("Which queue?", "Billing") == \
        "Which queue? The answer is Billing."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A two-layer model with the project's head, built offline."""
    name = "hf-internal-testing/tiny-random-BertModel"
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        config = AutoConfig.from_pretrained(name, num_labels=3)
        model = AutoModelForSequenceClassification.from_config(config)
    except Exception:                                            # noqa: BLE001
        pytest.skip("no local copy of the tiny test model")
    return Gavel(model, tokenizer, temperature=1.0, device="cpu", max_length=64)


def test_decide_returns_a_distribution(tiny):
    verdict = tiny.decide("a customer wrote in", "Which queue?",
                          ["Billing", "Access", "Fault"])
    assert verdict.option in ("Billing", "Access", "Fault")
    assert abs(sum(verdict.probabilities.values()) - 1.0) < 1e-4
    assert 0.0 <= verdict.confidence <= 1.0


def test_option_order_does_not_change_the_distribution(tiny):
    forward = tiny.decide("s", "q", ["alpha", "beta", "gamma"]).probabilities
    backward = tiny.decide("s", "q", ["gamma", "beta", "alpha"]).probabilities
    for option, value in forward.items():
        assert abs(value - backward[option]) < 1e-4


def test_threshold_marks_an_abstention(tiny):
    verdict = tiny.decide("s", "q", ["alpha", "beta"], threshold=1.01)
    assert verdict.abstained
