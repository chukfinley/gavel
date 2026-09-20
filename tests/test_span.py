"""The span scorer must do two things the marker head could not.

1. Point at the right tokens. If the spans are off by one the head pools the
   marker instead of the option and nothing downstream can be trusted.
2. Actually learn. The recorded failure of the first attempt was cross
   entropy stuck at ln(number of options) forever, so the test that matters
   is whether a handful of rows can be memorised at all.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.span import SpanScorer, build_tokenizer, encode

TINY = "google/bert_uncased_L-2_H-128_A-2"


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(TINY)


def test_spans_cover_the_option_text(tokenizer):
    rows = [("A charge appeared twice.", [("Which team?", ["billing", "tech"])])]
    batch = encode(tokenizer, rows, max_length=64)
    assert batch.option_mask.shape[1] == 2
    for slot, word in enumerate(["billing", "tech"]):
        picked = batch.input_ids[0][batch.option_mask[0, slot].bool()]
        text = tokenizer.decode(picked).strip()
        assert word in text.lower(), f"slot {slot} pooled {text!r}"
        # the marker itself must not be inside the span
        assert "[OPT]" not in text


def test_question_span_is_shared_by_its_options(tokenizer):
    rows = [("state", [("Which team?", ["a", "b"]), ("Urgent?", ["yes", "no"])])]
    batch = encode(tokenizer, rows, max_length=64)
    assert batch.group[0].tolist() == [0, 0, 1, 1]
    first = batch.question_mask[0, 0].bool()
    second = batch.question_mask[0, 1].bool()
    assert torch.equal(first, second)
    assert "which team" in tokenizer.decode(batch.input_ids[0][first]).lower()
    assert "urgent" in tokenizer.decode(batch.input_ids[0][batch.question_mask[0, 2].bool()]).lower()


def test_padding_slots_are_masked(tokenizer):
    rows = [("s", [("q", ["a", "b", "c"])]), ("s", [("q", ["a"])])]
    batch = encode(tokenizer, rows, max_length=64)
    assert batch.group[1].tolist() == [0, -1, -1]


def test_long_state_is_truncated_not_the_options(tokenizer):
    state = "word " * 4000
    rows = [(state, [("Which?", ["first option text", "second option text"])])]
    batch = encode(tokenizer, rows, max_length=128)
    assert batch.input_ids.shape[1] <= 128
    for slot, word in enumerate(["first", "second"]):
        text = tokenizer.decode(batch.input_ids[0][batch.option_mask[0, slot].bool()])
        assert word in text.lower()


def test_the_head_can_learn(tokenizer):
    """The test the marker head failed: memorise a few rows."""
    torch.manual_seed(0)
    model = SpanScorer(TINY, tokenizer)
    pairs = [
        ("I was charged twice for my subscription.", ["billing", "tech", "sales"], 0),
        ("The app crashes when I open a report.", ["billing", "tech", "sales"], 1),
        ("Do you give a discount for 50 seats?", ["billing", "tech", "sales"], 2),
        ("My invoice shows the wrong VAT number.", ["billing", "tech", "sales"], 0),
    ]
    rows = [(state, [("Which team?", options)]) for state, options, _ in pairs]
    labels = torch.tensor([label for _, _, label in pairs])
    batch = encode(tokenizer, rows, max_length=64)

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    start = None
    for step in range(60):
        scores = model(batch)
        loss = torch.nn.functional.cross_entropy(scores, labels)
        if start is None:
            start = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    chance = torch.log(torch.tensor(3.0)).item()
    assert start == pytest.approx(chance, abs=0.6), "should begin near chance"
    assert loss.item() < 0.1, (
        f"loss stuck at {loss.item():.3f}; chance is {chance:.3f}. "
        "This is the marker-head failure returning.")
    model.eval()
    with torch.no_grad():
        assert model(batch).argmax(dim=-1).tolist() == labels.tolist()


def test_group_softmax_normalises_per_question(tokenizer):
    model = SpanScorer(TINY, tokenizer)
    rows = [("state", [("Which team?", ["a", "b"]), ("Urgent?", ["yes", "no"])])]
    batch = encode(tokenizer, rows, max_length=64)
    with torch.no_grad():
        log_probabilities = model.group_log_softmax(model(batch), batch.group)
    probabilities = log_probabilities.exp()
    assert probabilities[0, :2].sum().item() == pytest.approx(1.0, abs=1e-4)
    assert probabilities[0, 2:].sum().item() == pytest.approx(1.0, abs=1e-4)


def test_reuse_encoder_actually_copies_the_weights(tokenizer, tmp_path):
    """Zero of thirty-nine tensors matched in the first version, silently.

    The pair checkpoint keys are `model.<base>.<layer>`, the span model's
    start at `<layer>`, and the token embedding differs by three marker rows.
    This pins that every layer arrives and the embedding rows are merged.
    """
    import importlib.util

    from gavel.model import EntailmentScorer

    spec = importlib.util.spec_from_file_location(
        "train_span", Path(__file__).resolve().parents[1] / "scripts" / "train_span.py")
    train_span = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_span)

    pair = EntailmentScorer(TINY)
    checkpoint = tmp_path / "pair.pt"
    torch.save({"model": pair.state_dict(), "backbone": TINY}, checkpoint)

    span = SpanScorer(TINY, tokenizer)
    reused, total = train_span.reuse_encoder(span.model, str(checkpoint))
    assert reused == total > 0

    source = pair.model.bert
    assert torch.equal(span.model.encoder.layer[0].attention.self.query.weight,
                       source.encoder.layer[0].attention.self.query.weight)
    rows = source.embeddings.word_embeddings.weight.shape[0]
    assert span.model.embeddings.word_embeddings.weight.shape[0] == rows + 3
    assert torch.equal(span.model.embeddings.word_embeddings.weight[:rows],
                       source.embeddings.word_embeddings.weight)
