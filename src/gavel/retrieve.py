"""Turn a knowledge question into a reading question.

The single largest measured gap in this project is not model size. It is
whether the answer is in the text the model was given:

    SciQ with its passage      0.980
    MedMCQA without a passage  0.282

Same model, same size. A 150 M encoder cannot store what a 4 B decoder read
during pretraining, but it does not have to: if a retrieval step puts the
relevant paragraphs into the state, recall becomes reading, and reading is
where this model already wins.

The index is built from the option texts and the question, scored with BM25 —
no embedding model, no second network, no GPU. That keeps the decision path
one forward pass plus a dictionary lookup.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


class Bm25:
    """A small BM25 index. Built once, queried per decision."""

    def __init__(self, passages: list[str], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.lengths = []
        self.frequencies: list[Counter] = []
        document_frequency: Counter = Counter()
        for passage in passages:
            words = tokens(passage)
            self.lengths.append(len(words))
            counted = Counter(words)
            self.frequencies.append(counted)
            document_frequency.update(counted.keys())
        self.average = sum(self.lengths) / max(len(self.lengths), 1)
        total = len(passages)
        self.idf = {word: math.log(1 + (total - count + 0.5) / (count + 0.5))
                    for word, count in document_frequency.items()}
        self.postings: dict[str, list[int]] = defaultdict(list)
        for index, counted in enumerate(self.frequencies):
            for word in counted:
                self.postings[word].append(index)

    def search(self, query: str, count: int = 3) -> list[str]:
        words = tokens(query)
        scores: dict[int, float] = defaultdict(float)
        for word in words:
            weight = self.idf.get(word)
            if weight is None:
                continue
            for index in self.postings[word]:
                frequency = self.frequencies[index][word]
                length = self.lengths[index]
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.average, 1e-6))
                scores[index] += weight * frequency * (self.k1 + 1) / denominator
        best = sorted(scores, key=scores.get, reverse=True)[:count]
        return [self.passages[i] for i in best]


def ground(decision, index: Bm25, passages: int = 3, joiner: str = "\n\n") -> None:
    """Put retrieved evidence in front of the state, in place.

    The query is the question together with every option, because the right
    paragraph often mentions one option and not the question's wording.
    """
    query = decision.question + " " + " ".join(o.description for o in decision.options)
    found = index.search(query, passages)
    if not found:
        return
    evidence = joiner.join(found)
    decision.state = (f"Evidence:\n{evidence}\n\n{decision.state}".strip()
                      if decision.state else f"Evidence:\n{evidence}")
