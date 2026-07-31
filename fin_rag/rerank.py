"""Reranking and HyDE query expansion — the two cheapest quality wins.

Both are optional (config flags) and both degrade gracefully, which is what
makes them safe to ship: if the reranker or the LLM is unavailable the
pipeline still returns fused results instead of erroring.
"""

from __future__ import annotations

from collections.abc import Sequence

from .embeddings import ngrams
from .llm import LLM
from .retrieval import Scored

HYDE_PROMPT = """你是一名金融研究助理。请针对下面的问题，写一段 2-3 句的假设性答案。
不需要保证事实准确，目的是生成与真实答案用词相似的文本，用于提升检索召回。

<QUESTION>
{question}
</QUESTION>
"""


def hyde_expand(llm: LLM, question: str) -> str:
    """Hypothetical Document Embeddings (Gao et al., 2022).

    A question and its answer often share few words ("能买吗" vs "建议增持").
    Embedding a *fake answer* instead of the question closes that lexical gap.
    Failure is non-fatal: we fall back to the raw question.
    """
    try:
        draft = llm.complete(HYDE_PROMPT.format(question=question))
    except Exception:  # pragma: no cover - provider/network failure path
        return ""
    draft = draft.strip()
    return "" if draft.startswith("根据现有资料无法回答") else draft


class LexicalOverlapReranker:
    """Dependency-free cross-encoder stand-in.

    A real deployment swaps in BAAI/bge-reranker-base (see `TorchReranker`).
    This scores token-level overlap with a length penalty, which is enough to
    demote chunks that only matched on one incidental term.
    """

    def __init__(self, *, length_penalty: float = 0.25) -> None:
        self.length_penalty = length_penalty

    def score(self, query: str, text: str) -> float:
        q_tokens = set(ngrams(query))
        d_tokens = ngrams(text)
        if not q_tokens or not d_tokens:
            return 0.0
        hits = sum(1 for t in d_tokens if t in q_tokens)
        coverage = len(q_tokens & set(d_tokens)) / len(q_tokens)
        density = hits / len(d_tokens)
        return float(coverage * (1 - self.length_penalty) + density * self.length_penalty)

    def rerank(self, query: str, results: Sequence[Scored], *, top_k: int) -> list[Scored]:
        if not results:
            return []
        rescored: list[Scored] = []
        for item in results:
            lex = self.score(query, item.chunk.index_text)
            # Blend rather than replace: the fused retrieval rank still carries
            # signal the lexical scorer cannot see.
            blended = 0.6 * lex + 0.4 * min(item.score * 10.0, 1.0)
            rescored.append(
                Scored(
                    chunk=item.chunk,
                    score=blended,
                    dense_rank=item.dense_rank,
                    sparse_rank=item.sparse_rank,
                )
            )
        rescored.sort(key=lambda s: -s.score)
        return rescored[:top_k]


class TorchReranker:  # pragma: no cover - requires optional heavy extra
    """Real cross-encoder. Same `.rerank()` contract as the lexical one."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: Sequence[Scored], *, top_k: int) -> list[Scored]:
        if not results:
            return []
        pairs = [(query, item.chunk.text) for item in results]
        scores = self._model.predict(pairs)
        rescored = [
            Scored(
                chunk=item.chunk,
                score=float(score),
                dense_rank=item.dense_rank,
                sparse_rank=item.sparse_rank,
            )
            for item, score in zip(results, scores, strict=False)
        ]
        rescored.sort(key=lambda s: -s.score)
        return rescored[:top_k]


def build_reranker(kind: str = "lexical"):
    if kind == "lexical":
        return LexicalOverlapReranker()
    if kind in {"bge", "cross-encoder"}:
        return TorchReranker()
    raise ValueError(f"unknown reranker: {kind!r}")
