"""Hybrid retrieval: dense vectors + sparse BM25, fused with RRF.

Why both: dense retrieval catches paraphrase ("回撤" vs "下跌幅度"), sparse
retrieval catches exact identifiers (fund codes, "2024Q3", ticker symbols)
that embeddings routinely blur. Finance queries are full of both.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder, cosine_scores, ngrams


@dataclass
class Scored:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class BM25:
    """Okapi BM25. Implemented directly (~30 lines) rather than pulled in as a
    dependency, so the ranking maths is inspectable and interview-defensible."""

    def __init__(self, corpus: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = [list(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.corpus else 0.0
        self.freqs: list[dict[str, int]] = []
        df: dict[str, int] = defaultdict(int)
        for doc in self.corpus:
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            self.freqs.append(counts)
            for token in counts:
                df[token] += 1
        n = len(self.corpus)
        # +0.5 smoothing keeps the idf of very common terms from going negative.
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def scores(self, query: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(self.corpus),), dtype=np.float32)
        if not self.corpus or self.avg_len == 0:
            return out
        for idx, counts in enumerate(self.freqs):
            length = self.doc_len[idx]
            total = 0.0
            for term in query:
                tf = counts.get(term)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                total += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            out[idx] = total
        return out


class HybridRetriever:
    def __init__(self, embedder: Embedder, *, rrf_k: int = 60) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self._matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self._bm25: BM25 | None = None

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            self._matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._bm25 = None
            return
        # index_text, not text: see Chunk.index_text for why the contextual
        # header matters. Both channels must see the same surface.
        self._matrix = self.embedder.encode([c.index_text for c in self.chunks])
        self._bm25 = BM25([ngrams(c.index_text) for c in self.chunks])

    def __len__(self) -> int:
        return len(self.chunks)

    def dense(self, query: str, *, k: int) -> list[int]:
        if not self.chunks:
            return []
        vec = self.embedder.encode([query])[0]
        scores = cosine_scores(vec, self._matrix)
        return _top_indices(scores, k)

    def sparse(self, query: str, *, k: int) -> list[int]:
        if not self.chunks or self._bm25 is None:
            return []
        scores = self._bm25.scores(ngrams(query))
        return _top_indices(scores, k)

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        candidate_k: int = 12,
        extra_queries: Sequence[str] = (),
    ) -> list[Scored]:
        """RRF over every (query, retriever) pair.

        RRF fuses by *rank*, not raw score, which is the point: BM25 scores are
        unbounded and cosine sits in [-1, 1], so summing them directly would
        let one channel dominate for arbitrary reasons.
        """
        if not self.chunks:
            return []
        queries = [query, *[q for q in extra_queries if q.strip()]]
        fused: dict[int, float] = defaultdict(float)
        dense_rank: dict[int, int] = {}
        sparse_rank: dict[int, int] = {}
        for q_idx, q in enumerate(queries):
            # Down-weight generated queries (HyDE) so they cannot outvote the
            # user's actual question.
            weight = 1.0 if q_idx == 0 else 0.5
            for rank, idx in enumerate(self.dense(q, k=candidate_k)):
                fused[idx] += weight / (self.rrf_k + rank + 1)
                dense_rank.setdefault(idx, rank)
            for rank, idx in enumerate(self.sparse(q, k=candidate_k)):
                fused[idx] += weight / (self.rrf_k + rank + 1)
                sparse_rank.setdefault(idx, rank)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            Scored(
                chunk=self.chunks[idx],
                score=float(score),
                dense_rank=dense_rank.get(idx),
                sparse_rank=sparse_rank.get(idx),
            )
            for idx, score in ordered
        ]


def _top_indices(scores: np.ndarray, k: int) -> list[int]:
    if scores.size == 0:
        return []
    k = min(k, scores.size)
    # argpartition is O(n) vs O(n log n) for a full sort; only the top-k slice
    # needs ordering afterwards.
    candidates = np.argpartition(-scores, k - 1)[:k]
    ranked = candidates[np.argsort(-scores[candidates])]
    return [int(i) for i in ranked if scores[int(i)] > 0]
