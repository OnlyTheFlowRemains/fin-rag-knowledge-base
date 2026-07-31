"""Embeddings.

Default backend is a dependency-free hashing vectoriser (char n-gram TF-IDF
projected into a fixed-width space). It is deterministic, needs no model
download, and is good enough to demonstrate dense retrieval on a small corpus.

`EMBEDDING_PROVIDER=sentence-transformers` swaps in a real semantic model
(BAAI/bge-small-zh-v1.5 by default) when the extra is installed. Same
interface, so nothing downstream changes.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def ngrams(text: str, *, n: int = 2) -> list[str]:
    """ASCII words plus CJK character n-grams.

    Chinese has no whitespace tokenisation, and character bigrams are a
    well-known strong baseline for Chinese IR — cheaper than loading jieba
    and robust to out-of-vocabulary finance jargon.
    """
    text = text.lower()
    tokens = [w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 1]
    for run in re.findall(r"[一-鿿]+", text):
        if len(run) < n:
            tokens.append(run)
            continue
        tokens.extend(run[i : i + n] for i in range(len(run) - n + 1))
    return tokens


class HashingEmbedder:
    """Signed hashing trick + sublinear TF, then L2 normalise."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        # Low bit decides the sign so unrelated tokens can cancel out instead
        # of always accumulating, which keeps collisions from inflating scores.
        return value % self.dim, 1.0 if value & 1 else -1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in ngrams(text):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                idx, sign = self._bucket(token)
                out[row, idx] += sign * (1.0 + math.log(count))
        return l2_normalise(out)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        from sentence_transformers import SentenceTransformer  # optional extra

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_embedder(provider: str, *, dim: int = 512) -> Embedder:
    if provider == "hashing":
        return HashingEmbedder(dim=dim)
    if provider in {"sentence-transformers", "bge"}:
        return SentenceTransformerEmbedder()
    raise ValueError(f"unknown embedding provider: {provider!r}")


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Both sides are L2-normalised, so a dot product *is* cosine similarity."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ query_vec.reshape(-1)


def mean_pool(vectors: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(list(vectors))
    pooled = stacked.mean(axis=0, keepdims=True)
    return l2_normalise(pooled)[0]
