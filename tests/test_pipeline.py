from __future__ import annotations

from fin_rag import MockLLM, RagPipeline
from fin_rag.chunking import Chunk
from fin_rag.compliance import Verdict
from fin_rag.embeddings import build_embedder
from fin_rag.evaluation import groundedness
from fin_rag.graph import build_context
from fin_rag.retrieval import HybridRetriever, Scored


def test_answer_cites_sources(pipeline):
    result = pipeline.answer("泓远沪深300增强ETF 的管理费年费率是多少")
    assert result.citations
    assert result.answer
    assert not result.refused
    assert all("source" in c for c in result.citations)


def test_answer_grounded_in_retrieved_context(pipeline):
    result = pipeline.answer("泓远稳健纯债债券型基金 2024 年最大回撤是多少")
    assert result.contexts
    # Anti-hallucination guarantee: nearly every token of the answer must be
    # traceable to a retrieved passage. Citation markers like "[2] " are the
    # only permitted additions, hence the 0.9 rather than 1.0 threshold.
    assert groundedness(result.answer, result.contexts) >= 0.9
    assert "1.12%" in result.answer


def test_compliance_refusal_skips_retrieval(pipeline):
    result = pipeline.answer("帮我下单买入 519301")
    assert result.refused
    assert result.verdict == Verdict.REFUSE.value
    assert result.citations == []
    assert "retrieve:" not in " ".join(result.trace)
    assert "refuse:compliance" in result.trace


def test_advice_question_answers_with_disclaimer(pipeline):
    result = pipeline.answer("泓远沪深300增强ETF 该不该买")
    assert result.verdict == Verdict.ALLOW_WITH_DISCLAIMER.value
    assert result.disclaimer
    assert "不构成投资建议" in result.disclaimer


def test_out_of_scope_question_refuses_instead_of_inventing():
    # empty index: the graph must route to refusal, never to generation
    retriever = HybridRetriever(build_embedder("hashing", dim=128))
    retriever.index([])
    pipe = RagPipeline(retriever=retriever, llm=MockLLM())
    result = pipe.answer("泓远沪深300增强ETF 的管理费率是多少")
    assert "无法回答" in result.answer
    assert result.citations == []
    assert "refuse:no-relevant-context" in result.trace


def test_answer_quotes_the_asked_about_fund_not_a_neighbour(pipeline):
    """Two funds both have a 费用结构 section. Answering with the other fund's
    number is the failure mode that matters most here, so pin it down."""
    hs300 = pipeline.answer("泓远沪深300增强ETF 的管理费年费率是多少")
    assert "0.50%" in hs300.answer
    assert "0.30%" not in hs300.answer

    bond = pipeline.answer("泓远稳健纯债债券型基金的管理费年费率")
    assert "0.30%" in bond.answer


def test_graph_trace_records_every_stage(pipeline):
    result = pipeline.answer("投资者风险承受能力如何分级")
    joined = " ".join(result.trace)
    for stage in ("compliance:", "expand:", "retrieve:", "rerank:", "generate:"):
        assert stage in joined, stage


def test_build_context_numbers_and_dedupes():
    parent = "共享的上下文段落。"
    hits = [
        Scored(chunk=Chunk(doc_id="d", chunk_id="d#0", text="甲。", context=parent), score=0.5),
        Scored(chunk=Chunk(doc_id="d", chunk_id="d#1", text="乙。", context=parent), score=0.4),
        Scored(chunk=Chunk(doc_id="e", chunk_id="e#0", text="丙。", context="另一段。"), score=0.3),
    ]
    context, citations = build_context(hits)
    assert context.count(parent) == 1
    assert [c["index"] for c in citations] == [1, 2]


def test_disclaimer_suppressed_when_nothing_retrieved():
    retriever = HybridRetriever(build_embedder("hashing", dim=128))
    retriever.index([])
    pipe = RagPipeline(retriever=retriever, llm=MockLLM())
    result = pipe.answer("这只基金该不该买")
    # no sources means no product information was given, so a product
    # disclaimer would be misleading noise
    assert result.disclaimer == ""


def test_hyde_failure_does_not_break_pipeline(pipeline, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(pipeline.llm, "complete", boom)
    result = pipeline.answer("管理费年费率是多少")
    # Provider is down for both HyDE and generation. The request must degrade
    # to "here are your sources" rather than raise.
    assert "生成服务暂时不可用" in result.answer
    assert result.citations, "retrieved sources must survive a generation failure"
    assert any(r.startswith("generation-error:") for r in result.reasons)
    assert "generate:failed" in result.trace
