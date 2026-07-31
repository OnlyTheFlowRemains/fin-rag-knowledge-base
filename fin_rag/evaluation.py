"""Offline evaluation — RAGAS-style metrics without the dependency.

"It feels better" is not an answer in an interview. These four metrics are the
minimum needed to say whether a change (chunk size, HyDE on/off, reranker)
actually helped, and all of them are computable from a small labelled set with
no LLM judge, so they run in CI.

- hit_rate            did any expected doc make the top-k
- mrr                 how high the first correct doc ranked
- context_precision   fraction of returned chunks that were relevant
- groundedness        fraction of answer tokens traceable to the context
                      (a cheap faithfulness/hallucination proxy)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .embeddings import ngrams
from .graph import RagPipeline


@dataclass
class EvalCase:
    question: str
    expected_doc_ids: list[str]
    must_refuse: bool = False


@dataclass
class EvalReport:
    n: int
    hit_rate: float
    mrr: float
    context_precision: float
    groundedness: float
    refusal_accuracy: float
    per_case: list[dict[str, object]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"cases={self.n} hit_rate={self.hit_rate:.3f} mrr={self.mrr:.3f} "
            f"ctx_precision={self.context_precision:.3f} "
            f"groundedness={self.groundedness:.3f} refusal_acc={self.refusal_accuracy:.3f}"
        )


def _doc_root(doc_id: str) -> str:
    """'fund-a::s2' -> 'fund-a'. The section suffix is an indexing detail."""
    return doc_id.split("::")[0]


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    answer_tokens = ngrams(answer)
    if not answer_tokens:
        return 0.0
    supported = set()
    for ctx in contexts:
        supported.update(ngrams(ctx))
    covered = sum(1 for t in answer_tokens if t in supported)
    return covered / len(answer_tokens)


def evaluate(pipeline: RagPipeline, cases: Sequence[EvalCase]) -> EvalReport:
    if not cases:
        return EvalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    hits = 0.0
    reciprocal = 0.0
    precision_total = 0.0
    ground_total = 0.0
    refusal_correct = 0.0
    rows: list[dict[str, object]] = []

    for case in cases:
        result = pipeline.answer(case.question)
        retrieved = [_doc_root(str(c["doc_id"])) for c in result.citations]
        expected = set(case.expected_doc_ids)

        if case.must_refuse:
            correct = result.refused or "无法回答" in result.answer
            refusal_correct += 1.0 if correct else 0.0
            rows.append({"question": case.question, "refused": result.refused, "ok": correct})
            continue

        refusal_correct += 1.0 if not result.refused else 0.0
        matched = [d for d in retrieved if d in expected]
        if matched:
            hits += 1.0
            first = next(i for i, d in enumerate(retrieved) if d in expected)
            reciprocal += 1.0 / (first + 1)
        if retrieved:
            precision_total += len(matched) / len(retrieved)
        ground = groundedness(result.answer, result.contexts)
        ground_total += ground
        rows.append(
            {
                "question": case.question,
                "retrieved": retrieved,
                "expected": sorted(expected),
                "hit": bool(matched),
                "groundedness": round(ground, 3),
            }
        )


    n = len(cases)
    answered = max(1, sum(1 for c in cases if not c.must_refuse))
    return EvalReport(
        n=n,
        hit_rate=hits / answered,
        mrr=reciprocal / answered,
        context_precision=precision_total / answered,
        groundedness=ground_total / answered,
        refusal_accuracy=refusal_correct / n,
        per_case=rows,
    )


def compare_configs(
    build_pipeline, cases: Sequence[EvalCase], variants: dict[str, dict[str, bool]]
) -> dict[str, EvalReport]:
    """Run the same cases across config variants — this is how you justify
    "HyDE gained us X" with a number instead of a vibe."""
    return {name: evaluate(build_pipeline(**kwargs), cases) for name, kwargs in variants.items()}
