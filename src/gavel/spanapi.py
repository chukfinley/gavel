"""The caller-facing side of the one-sequence model.

    from gavel.spanapi import SpanGavel

    judge = SpanGavel.from_checkpoint("runs/span/best.pt")
    judge.decide(state, "Which queue?", ["Billing", "Access", "Fault"])
    judge.decide_many(state, [("Which queue?", [...]), ("Urgent?", ["Yes", "No"])])

`decide` keeps the signature of `Gavel.decide`, so every evaluation script
works against either model. `decide_many` is the reason the architecture was
changed: several questions about one state cost one reading of the state.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .api import Verdict
from .span import SpanScorer, build_tokenizer, encode


class SpanGavel:
    """A decision model that reads the state once."""

    ABSTAIN = "The information given does not decide this"

    def __init__(self, model: SpanScorer, tokenizer, device: str | None = None,
                 max_length: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None,
                        max_length: int = 1024) -> SpanGavel:
        state = torch.load(path, map_location="cpu", weights_only=False)
        tokenizer = build_tokenizer(state["backbone"])
        model = SpanScorer(state["backbone"], tokenizer)
        model.load_state_dict(state["model"])
        return cls(model, tokenizer, device, max_length)

    @torch.no_grad()
    def decide_many(self, state: str,
                    questions: Sequence[tuple[str, Sequence[str]]]) -> list[Verdict]:
        """One reading of the state, one verdict per question."""
        rows = [(state, [(q, list(o)) for q, o in questions])]
        batch = encode(self.tokenizer, rows, self.max_length, self.device)
        scores = self.model(batch)
        log_probabilities = self.model.group_log_softmax(scores, batch.group)
        probabilities = log_probabilities.exp()[0]
        group = batch.group[0]

        verdicts = []
        for index, (question, options) in enumerate(questions):
            slots = (group == index).nonzero(as_tuple=True)[0]
            values = [float(probabilities[slot]) for slot in slots]
            best = max(range(len(values)), key=values.__getitem__)
            verdicts.append(Verdict(
                option=list(options)[best], confidence=round(values[best], 4),
                probabilities={str(o): round(v, 4)
                               for o, v in zip(options, values)}))
        return verdicts

    def decide(self, state: str, question: str, options: Sequence[str],
               abstain: bool = False, threshold: float = 0.0) -> Verdict:
        choices = list(options) + ([self.ABSTAIN] if abstain else [])
        verdict = self.decide_many(state, [(question, choices)])[0]
        if abstain and verdict.option == self.ABSTAIN:
            verdict.abstained = True
        if threshold and verdict.confidence < threshold:
            verdict.abstained = True
        return verdict
