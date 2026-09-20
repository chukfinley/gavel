"""The fast index must return what the plain implementation returns.

The rewrite exists because the loop version stalled the grounded build at
5000 of 40000 rows. Speed is worthless if the passages change, so this
scores a reference BM25 written the obvious way and compares.
"""

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gavel.retrieve import Bm25, ground
from gavel.schema import Decision, Option

WORD = re.compile(r"[a-z0-9]+")

PASSAGES = [
    "Mercury: the smallest planet in the solar system and the closest to the Sun.",
    "Venus: the second planet from the Sun, with a thick carbon dioxide atmosphere.",
    "Earth: the third planet from the Sun and the only one known to hold life.",
    "Mars: the fourth planet, often called the red planet because of iron oxide.",
    "Jupiter: the largest planet in the solar system, a gas giant with a great red spot.",
    "Photosynthesis: plants convert light into chemical energy stored as sugar.",
    "Mitochondria: the organelles that produce most of a cell's chemical energy.",
    "The Rhine: a river that runs from the Alps to the North Sea through Germany.",
    "Penicillin: an antibiotic discovered by Alexander Fleming in 1928.",
    "Gravity: the attraction between masses, described by general relativity.",
]


def reference(passages, query, count, k1=1.5, b=0.75):
    """BM25 written the obvious way, as the thing to match."""
    lengths, frequencies = [], []
    document_frequency = Counter()
    for passage in passages:
        words = WORD.findall(passage.lower())
        lengths.append(len(words))
        counted = Counter(words)
        frequencies.append(counted)
        document_frequency.update(counted.keys())
    average = sum(lengths) / len(lengths)
    total = len(passages)
    idf = {w: math.log(1 + (total - c + 0.5) / (c + 0.5))
           for w, c in document_frequency.items()}
    scores = defaultdict(float)
    for word in set(WORD.findall(query.lower())):
        weight = idf.get(word)
        if weight is None:
            continue
        for index, counted in enumerate(frequencies):
            frequency = counted.get(word, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (1 - b + b * lengths[index] / average)
            scores[index] += weight * frequency * (k1 + 1) / denominator
    best = sorted(scores, key=lambda i: (-scores[i], i))[:count]
    return [passages[i] for i in best]


@pytest.mark.parametrize("query", [
    "which planet is closest to the sun",
    "what produces energy in a cell",
    "who discovered penicillin",
    "the largest planet gas giant",
    "river through germany to the north sea",
])
def test_matches_the_reference(query):
    index = Bm25(PASSAGES)
    assert index.search(query, 3) == reference(PASSAGES, query, 3)


def test_unknown_words_give_nothing():
    index = Bm25(PASSAGES)
    assert index.search("zzzz qqqq", 3) == []


def test_asks_for_more_than_it_can_give():
    index = Bm25(PASSAGES)
    found = index.search("penicillin", 5)
    assert 1 <= len(found) <= 5
    assert "Penicillin" in found[0]


def test_ground_puts_evidence_before_the_state():
    index = Bm25(PASSAGES)
    decision = Decision(id="1", state="A pub quiz answer sheet.",
                        question="Which planet is closest to the Sun?",
                        options=[Option("a", "Mercury"), Option("b", "Mars")],
                        label=0)
    ground(decision, index, passages=2)
    assert decision.state.startswith("Evidence:")
    assert "A pub quiz answer sheet." in decision.state
    assert "Mercury" in decision.state


def test_is_fast_enough_to_build_the_corpus():
    """The loop version stalled here; 2000 passages must stay interactive."""
    import time

    passages = [f"{PASSAGES[i % len(PASSAGES)]} Document number {i} of the set."
                for i in range(2000)]
    index = Bm25(passages)
    started = time.perf_counter()
    for _ in range(50):
        index.search("which planet is closest to the sun and the smallest", 3)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"50 queries took {elapsed:.2f}s"
