"""The option scatter must survive bf16 autocast.

The optimised run died at its first dev evaluation with 'Index put requires
the source and destination dtypes match, got BFloat16 for the destination
and Float for the source': under autocast the logits come back in bf16, the
destination was built in that dtype, and dividing by the fp32 temperature
promoted the source. The training loop cast to float first; the evaluation
path did not. CPU autocast in bf16 reproduces it here without a card.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.encoding import encode_options
from gavel.model import EntailmentScorer, build_tokenizer
from gavel.schema import Decision, Option

TINY = "google/bert_uncased_L-2_H-128_A-2"


def _decisions():
    return [
        Decision(id="a", state="I was charged twice.", question="Which team?",
                 options=[Option("b", "billing"), Option("t", "tech")], label=0),
        Decision(id="b", state="The app crashes on launch.", question="Which team?",
                 options=[Option("b", "billing"), Option("t", "tech"),
                          Option("s", "sales")], label=1),
    ]


def test_option_logits_under_bf16_autocast():
    model = EntailmentScorer(TINY).eval()
    tokenizer = build_tokenizer(TINY)
    encoding, mask, _ = encode_options(_decisions(), tokenizer, 64)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        scores = model.option_logits(dict(encoding), mask)
    assert scores.shape == (2, 3)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores[mask.bool()]).all()
    assert scores[0, 2] == torch.finfo(torch.float32).min      # the empty slot


def test_scatter_matches_a_plain_loop():
    model = EntailmentScorer(TINY).eval()
    tokenizer = build_tokenizer(TINY)
    decisions = _decisions()
    encoding, mask, _ = encode_options(decisions, tokenizer, 64)
    with torch.no_grad():
        flat = model.pair_logits(dict(encoding))[:, 0]
        scores = model.scatter_options(flat, mask)
    position = 0
    for row, decision in enumerate(decisions):
        for column in range(len(decision.options)):
            assert torch.isclose(scores[row, column], flat[position].float()
                                 / model.log_temperature.exp())
            position += 1
    assert position == flat.shape[0]                            # no filler pairs
