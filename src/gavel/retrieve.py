"""Turn a knowledge question into a reading question.

The single largest measured gap in this project is not model size. It is
whether the answer is in the text the model was given:

    SciQ with its passage      0.980
    MedMCQA without a passage  0.282

Same model, same size. A 150 M encoder cannot store what a 4 B decoder read
during pretraining, but it does not have to: if a retrieval step puts the
relevant paragraphs into the state, recall becomes reading, and reading is
where this model already wins.

The index is BM25 over the passages, scored without an embedding model, a
second network or a GPU, so the decision path stays one forward pass plus a
lookup.

## Why this is a matrix and not a loop

The first version walked a posting list per query term in Python. That is
fine until a query contains an ordinary word: "the" or "is" appears in most
of 120 000 passages, so one term costs 100 000 dictionary lookups, and
building the grounded training set stalled at 5000 of 40000 rows.

The weights do not depend on the query, only on the passage and the term, so
all of them are computed once into a sparse matrix. A query is then a column
selection and a row sum, which SciPy does in C. Same scores, three orders of
magnitude less Python.
"""

from __future__ import annotations

import re

import numpy as np

WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


class Bm25:
    """A BM25 index with the document weights precomputed.

    `search` returns the same passages the straightforward implementation
    returns; `tests/test_retrieve.py` checks that against a reference.
    """

    def __init__(self, passages: list[str], k1: float = 1.5, b: float = 0.75):
        from scipy import sparse
        from sklearn.feature_extraction.text import CountVectorizer

        self.passages = passages
        self.k1, self.b = k1, b
        self.vectorizer = CountVectorizer(tokenizer=tokens, lowercase=True,
                                          token_pattern=None, dtype=np.float32)
        counts = self.vectorizer.fit_transform(passages).tocsc()
        self.vocabulary = self.vectorizer.vocabulary_

        total = counts.shape[0]
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        average = float(lengths.mean()) if total else 1.0
        document_frequency = np.diff(counts.indptr)
        idf = np.log(1 + (total - document_frequency + 0.5)
                     / (document_frequency + 0.5)).astype(np.float32)

        # weight(document, term) = idf * tf (k1+1) / (tf + k1 (1-b+b len/avg))
        frequency = counts.data
        rows = counts.indices
        norm = self.k1 * (1 - self.b + self.b * lengths[rows] / max(average, 1e-6))
        weighted = frequency * (self.k1 + 1) / (frequency + norm)
        term_of = np.repeat(np.arange(counts.shape[1]), np.diff(counts.indptr))
        weighted = weighted * idf[term_of]

        self.matrix = sparse.csc_matrix(
            (weighted.astype(np.float32), counts.indices, counts.indptr),
            shape=counts.shape)

    def search(self, query: str, count: int = 3) -> list[str]:
        columns = [self.vocabulary[word] for word in set(tokens(query))
                   if word in self.vocabulary]
        if not columns:
            return []
        scores = np.asarray(self.matrix[:, columns].sum(axis=1)).ravel()
        if not scores.any():
            return []
        count = min(count, int((scores > 0).sum()))
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        return [self.passages[int(i)] for i in top]


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
