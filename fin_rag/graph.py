"""The RAG workflow as an explicit LangGraph state machine.

Why a graph instead of a linear chain: the flow genuinely branches. A
compliance refusal must skip retrieval entirely; an empty/low-confidence
retrieval must skip generation and return a refusal. Encoding that as a graph
makes every path testable in isolation and keeps the trace readable.

    compliance ─(refuse)────────────────────────────► finalise
         │
      (allow)
         ▼
      expand ──► retrieve ──► rerank ─(no hit)─────► finalise
                                 │
                              (hit)
                                 ▼
                             generate ──────────────► finalise
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, TypedDict

from langgraph.graph import END, StateGraph

from .chunking import Chunk
from .compliance import REFUSAL_MESSAGE, Verdict, check_query, sanitise_answer
from .config import Settings, load_settings
from .llm import LLM
from .rerank import build_reranker, hyde_expand
from .retrieval import HybridRetriever, Scored

ANSWER_PROMPT = """你是一名严谨的金融知识库助理。只能依据 <CONTEXT> 中的内容回答问题。

规则：
1. 不得编造 <CONTEXT> 里没有的数字、日期或结论。
2. 无法从 <CONTEXT> 得出答案时，直接回答"根据现有资料无法回答该问题"。
3. 不得给出个人投资建议，不得预测价格，不得使用"保证收益""稳赚"等表述。
4. 引用事实时标注来源编号，如 [1]。

<CONTEXT>
{context}
</CONTEXT>

<QUESTION>
{question}
</QUESTION>
"""

GENERATION_FAILED_MESSAGE = (
    "生成服务暂时不可用，以下为检索到的相关文档来源，请直接查阅原文。"
)


class RagState(TypedDict, total=False):
    question: str
    hyde: str
    hits: list[Scored]
    answer: str
    contexts: list[str]
    citations: list[dict[str, Any]]
    verdict: str
    reasons: list[str]
    disclaimer: str
    trace: list[str]


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    # Raw passages sent to the model. Kept so groundedness can be scored and so
    # a reviewer can see exactly what the answer was allowed to rely on.
    contexts: list[str] = field(default_factory=list)
    verdict: str = Verdict.ALLOW.value
    reasons: list[str] = field(default_factory=list)
    disclaimer: str = ""
    trace: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.verdict == Verdict.REFUSE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "contexts": self.contexts,
            "verdict": self.verdict,
            "reasons": self.reasons,
            "disclaimer": self.disclaimer,
            "trace": self.trace,
        }


def build_context(hits: Sequence[Scored]) -> tuple[str, list[dict[str, Any]]]:
    """Numbered context blocks + parallel citation records.

    Deduplicated on `context` because small_to_big children share a parent
    window; sending the same passage three times just burns tokens.
    """
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        passage = hit.chunk.context
        if passage in seen:
            continue
        seen.add(passage)
        index = len(blocks) + 1
        blocks.append(f"[{index}] {passage}")
        citations.append(
            {
                "index": index,
                "doc_id": hit.chunk.doc_id,
                "chunk_id": hit.chunk.chunk_id,
                "source": hit.chunk.citation,
                "score": round(hit.score, 4),
                "dense_rank": hit.dense_rank,
                "sparse_rank": hit.sparse_rank,
            }
        )
    return "\n\n".join(blocks), citations


class RagPipeline:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        llm: LLM,
        settings: Settings | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings or load_settings()
        self.reranker = reranker or build_reranker("lexical")
        self._graph = self._compile()

    # ---------------- nodes ----------------

    def _node_compliance(self, state: RagState) -> RagState:
        result = check_query(state["question"])
        return {
            "verdict": result.verdict.value,
            "reasons": result.reasons,
            "disclaimer": result.disclaimer,
            "trace": [*state.get("trace", []), f"compliance:{result.verdict.value}"],
        }

    def _node_expand(self, state: RagState) -> RagState:
        if not self.settings.retrieval.use_hyde:
            return {"hyde": "", "trace": [*state.get("trace", []), "expand:skipped"]}
        draft = hyde_expand(self.llm, state["question"])
        return {
            "hyde": draft,
            "trace": [*state.get("trace", []), f"expand:hyde({len(draft)}chars)"],
        }

    def _node_retrieve(self, state: RagState) -> RagState:
        cfg = self.settings.retrieval
        extra = [state.get("hyde", "")] if state.get("hyde") else []
        hits = self.retriever.search(
            state["question"],
            top_k=cfg.candidate_k,
            candidate_k=cfg.candidate_k,
            extra_queries=extra,
        )
        return {"hits": hits, "trace": [*state.get("trace", []), f"retrieve:{len(hits)}"]}

    def _node_rerank(self, state: RagState) -> RagState:
        cfg = self.settings.retrieval
        hits = state.get("hits", [])
        if cfg.use_rerank and hits:
            hits = self.reranker.rerank(state["question"], hits, top_k=cfg.top_k)
            label = f"rerank:{len(hits)}"
        else:
            hits = hits[: cfg.top_k]
            label = "rerank:skipped"
        hits = [h for h in hits if h.score >= cfg.min_score]
        return {"hits": hits, "trace": [*state.get("trace", []), f"{label}->{len(hits)}"]}

    def _node_generate(self, state: RagState) -> RagState:
        context, citations = build_context(state.get("hits", []))
        prompt = ANSWER_PROMPT.format(context=context, question=state["question"])
        try:
            raw = self.llm.complete(prompt)
        except Exception as exc:  # provider timeout / rate limit / network
            # Degrade, do not 500. The retrieved sources are still useful to
            # the caller, and an upstream hiccup should not lose them.
            return {
                "answer": GENERATION_FAILED_MESSAGE,
                "contexts": [h.chunk.context for h in state.get("hits", [])],
                "citations": citations,
                "reasons": [*state.get("reasons", []), f"generation-error:{type(exc).__name__}"],
                "trace": [*state.get("trace", []), "generate:failed"],
            }
        answer, notes = sanitise_answer(raw)
        return {
            "answer": answer.strip(),
            "contexts": [h.chunk.context for h in state.get("hits", [])],
            "citations": citations,
            "reasons": [*state.get("reasons", []), *notes],
            "trace": [*state.get("trace", []), f"generate:{len(answer)}chars"],
        }

    def _node_refuse(self, state: RagState) -> RagState:
        if state.get("verdict") == Verdict.REFUSE.value:
            message = REFUSAL_MESSAGE
            reason = "refuse:compliance"
        else:
            message = "根据现有资料无法回答该问题。"
            reason = "refuse:no-relevant-context"
        return {
            "answer": message,
            "citations": [],
            "contexts": [],
            "trace": [*state.get("trace", []), reason],
        }

    # ---------------- edges ----------------

    @staticmethod
    def _route_compliance(state: RagState) -> str:
        return "refuse" if state.get("verdict") == Verdict.REFUSE.value else "expand"

    @staticmethod
    def _route_hits(state: RagState) -> str:
        return "generate" if state.get("hits") else "refuse"

    def _compile(self):
        graph = StateGraph(RagState)
        graph.add_node("compliance", self._node_compliance)
        graph.add_node("expand", self._node_expand)
        graph.add_node("retrieve", self._node_retrieve)
        graph.add_node("rerank", self._node_rerank)
        graph.add_node("generate", self._node_generate)
        graph.add_node("refuse", self._node_refuse)
        graph.set_entry_point("compliance")
        graph.add_conditional_edges(
            "compliance", self._route_compliance, {"refuse": "refuse", "expand": "expand"}
        )
        graph.add_edge("expand", "retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_conditional_edges(
            "rerank", self._route_hits, {"generate": "generate", "refuse": "refuse"}
        )
        graph.add_edge("generate", END)
        graph.add_edge("refuse", END)
        return graph.compile()

    # ---------------- public API ----------------

    def answer(self, question: str) -> RagAnswer:
        final: RagState = self._graph.invoke({"question": question, "trace": []})
        disclaimer = final.get("disclaimer", "")
        verdict = final.get("verdict", Verdict.ALLOW.value)
        if verdict == Verdict.REFUSE.value or not final.get("citations"):
            disclaimer = ""
        return RagAnswer(
            question=question,
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            contexts=final.get("contexts", []),
            verdict=verdict,
            reasons=final.get("reasons", []),
            disclaimer=disclaimer,
            trace=final.get("trace", []),
        )


def index_chunks(retriever: HybridRetriever, chunks: Sequence[Chunk]) -> HybridRetriever:
    retriever.index(chunks)
    return retriever
