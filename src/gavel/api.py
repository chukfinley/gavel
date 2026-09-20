"""The library a caller uses.

    from gavel import Gavel

    judge = Gavel.from_pretrained("chukfinley/gavel-vela-32k")
    judge.decide(
        state="The customer was charged twice for the same subscription period.",
        question="Which queue should handle this request?",
        options=["Billing support", "Account access support", "Technical fault"],
    )
    # Verdict(option='Billing support', confidence=0.83, probabilities={...})

Everything is one forward pass per option, no text is generated, and the
probabilities are temperature-corrected with the value fitted at training time.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .encoding import CLAIM_PREFIX

ENTAILMENT = 0


@dataclass
class Verdict:
    option: str
    confidence: float
    probabilities: dict[str, float]
    abstained: bool = False
    meta: dict = field(default_factory=dict)


class Gavel:
    """A decision model: state and options in, one typed answer out."""

    ABSTAIN = "The information given does not decide this"

    def __init__(self, model, tokenizer, temperature: float = 1.0,
                 device: str | None = None, max_length: int = 1024):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature or 1.0
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------ load
    @classmethod
    def from_pretrained(cls, name: str, device: str | None = None,
                        max_length: int = 1024) -> Gavel:
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        config = AutoConfig.from_pretrained(name)
        settings = getattr(config, "gavel", None) or {}
        return cls(model, tokenizer, float(settings.get("temperature", 1.0)),
                   device, max_length)

    @classmethod
    def from_checkpoint(cls, path: str, device: str | None = None,
                        max_length: int = 1024) -> Gavel:
        """Load a training checkpoint produced by scripts/train.py."""
        from .model import EntailmentScorer, build_tokenizer

        state = torch.load(path, map_location="cpu", weights_only=False)
        scorer = EntailmentScorer(state["backbone"])
        scorer.load_state_dict(state["model"])
        return cls(scorer.model, build_tokenizer(state["backbone"]),
                   float(scorer.log_temperature.exp()), device, max_length)

    # ----------------------------------------------------------------- score
    def _hypothesis(self, question: str, option: str) -> str:
        question = question.strip()
        option = option.strip()
        if question.startswith(CLAIM_PREFIX):
            return f"{question[len(CLAIM_PREFIX):].rstrip('.')}. {option}."
        return f"{question} The answer is {option}."

    @torch.no_grad()
    def decide(self, state: str, question: str, options: Sequence[str],
               abstain: bool = False, threshold: float = 0.0) -> Verdict:
        """Pick one option. With `abstain`, an extra option is offered."""
        choices = list(options) + ([self.ABSTAIN] if abstain else [])
        premise = (state or question).strip()
        pairs = [self._hypothesis(question, choice) for choice in choices]
        encoding = self.tokenizer([premise] * len(pairs), pairs, padding=True,
                                  truncation=True, max_length=self.max_length,
                                  return_tensors="pt").to(self.device)
        # bf16 autocast on a card: the model was trained and calibrated that
        # way, and flash-attention accepts nothing else. Without this the
        # pod's benchmark evaluations crashed and published no numbers.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=str(self.device).startswith("cuda")):
            logits = self.model(**encoding).logits[:, ENTAILMENT].float() / self.temperature
        probabilities = F.softmax(logits, dim=-1).cpu().tolist()
        best = max(range(len(choices)), key=probabilities.__getitem__)
        confidence = probabilities[best]
        abstained = abstain and choices[best] == self.ABSTAIN
        if threshold and confidence < threshold:
            abstained = True
        return Verdict(
            option=choices[best], confidence=round(confidence, 4),
            probabilities={choice: round(p, 4) for choice, p in zip(choices, probabilities)},
            abstained=abstained)

    def decide_many(self, state: str, questions: Iterable[dict]) -> list[Verdict]:
        """Several questions about the same state.

        The state is encoded again for every option, therefore this is a
        convenience, not a saving. A head that reads one state and many
        questions at once is the open piece of work; see HANDOVER.md.
        """
        return [self.decide(state, q["question"], q["options"],
                            abstain=q.get("abstain", False),
                            threshold=q.get("threshold", 0.0)) for q in questions]

    @torch.no_grad()
    def decide_long(self, state: str, question: str, options: Sequence[str],
                    window: int = 1400, overlap: int = 200) -> Verdict:
        """For a state longer than the window: score overlapping pieces."""
        from .longdoc import pieces

        windows = pieces(state, self.tokenizer, window, overlap)
        if len(windows) == 1:
            return self.decide(state, question, options)
        pairs = [(piece, self._hypothesis(question, choice))
                 for choice in options for piece in windows]
        scores = []
        for start in range(0, len(pairs), 8):
            chunk = pairs[start : start + 8]
            encoding = self.tokenizer([a for a, _ in chunk], [b for _, b in chunk],
                                      padding=True, truncation=True,
                                      max_length=window + 128, return_tensors="pt").to(self.device)
            scores.append(self.model(**encoding).logits[:, ENTAILMENT].float().cpu())
        matrix = torch.cat(scores).view(len(options), len(windows))
        best_per_option = matrix.max(dim=1).values / self.temperature
        probabilities = F.softmax(best_per_option, dim=-1).tolist()
        best = max(range(len(options)), key=probabilities.__getitem__)
        return Verdict(
            option=options[best], confidence=round(probabilities[best], 4),
            probabilities={o: round(p, 4) for o, p in zip(options, probabilities)},
            meta={"windows": len(windows)})
