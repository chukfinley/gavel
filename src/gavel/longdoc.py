"""Decisions over documents that are longer than the window.

The backbone reads 8192 tokens; Jev advertises 32000. Raising the window
itself is not an option here — the position encoding was pretrained at 8192
and the global attention layers cost time and memory quadratically.

The way out follows from the architecture: every option is already scored in
its own forward pass, therefore the state can be cut into overlapping pieces
and each option scored against each piece. Cost then grows linearly with the
document, memory stays flat, and the length is bounded only by patience.

Aggregation over the pieces is a max, because entailment is existential: if
one piece establishes the claim, the claim holds, and a piece that says
nothing about it should not drag the score down. A mean would do exactly that
and would punish long documents.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .encoding import hypothesis, premise
from .model import CONTRADICTION, ENTAILMENT
from .schema import Decision


def pieces(text: str, tokenizer, size: int = 1400, overlap: int = 200) -> list[str]:
    """Overlapping windows, so that a sentence on a boundary is not lost."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= size:
        return [text]
    step = max(size - overlap, 1)
    return [tokenizer.decode(ids[start : start + size])
            for start in range(0, len(ids), step)
            if start < len(ids)]


@torch.no_grad()
def decide(model, tokenizer, decision: Decision, device="cuda",
           size: int = 1400, overlap: int = 200, batch_size: int = 8) -> dict:
    """Probabilities over the options, for a state of any length."""
    windows = pieces(premise(decision), tokenizer, size, overlap)
    hypotheses = [hypothesis(decision, index) for index in range(len(decision.options))]

    pairs = [(window, text) for text in hypotheses for window in windows]
    entail, contra = [], []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        encoding = tokenizer([left for left, _ in chunk], [right for _, right in chunk],
                             padding=True, truncation=True, max_length=size + 128,
                             return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            logits = model.pair_logits(dict(encoding)).float()
        entail.append(logits[:, ENTAILMENT].cpu())
        contra.append(logits[:, CONTRADICTION].cpu())

    width = len(windows)
    entail = torch.cat(entail).view(len(hypotheses), width)
    contra = torch.cat(contra).view(len(hypotheses), width)

    # One supporting piece is enough; one contradicting piece is enough as well,
    # and it is subtracted so that a document which refutes an option cannot win
    # on the strength of an unrelated passage.
    score = entail.max(dim=1).values - 0.5 * contra.max(dim=1).values
    score = score / float(model.log_temperature.exp())
    probabilities = F.softmax(score, dim=-1)
    best = int(probabilities.argmax())
    return {
        "option": decision.options[best].id,
        "probabilities": {option.id: round(float(p), 4)
                          for option, p in zip(decision.options, probabilities)},
        "confidence": round(float(probabilities[best]), 4),
        "windows": width,
        "state_tokens": len(tokenizer.encode(premise(decision), add_special_tokens=False)),
    }
