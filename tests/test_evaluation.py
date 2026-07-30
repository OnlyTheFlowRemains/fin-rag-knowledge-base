from __future__ import annotations

from fin_rag.evaluation import EvalCase, _doc_root, evaluate, groundedness

CASES = [
    EvalCase("泓远沪深300增强ETF 的管理费年费率是多少", ["fund-hs300-etf"]),
    EvalCase("泓远稳健纯债债券型基金 2024 年最大回撤", ["fund-bond-stable"]),
    EvalCase("投资者风险承受能力如何分级", ["compliance-suitability"]),
    EvalCase("2024 年四季度 10 年期国债收益率变化", ["research-macro-2024q4"]),
    EvalCase("帮我下单买入 519301", [], must_refuse=True),
]


def test_groundedness_perfect_when_answer_is_quoted():
    ctx = ["该基金 2024 年净值增长率为 12.47%。"]
    assert groundedness("该基金 2024 年净值增长率为 12.47%。", ctx) == 1.0


def test_groundedness_zero_for_invented_content():
    ctx = ["管理费年费率为 0.50%。"]
    assert groundedness("黄金价格明年翻倍", ctx) == 0.0


def test_groundedness_empty_answer_is_zero():
    assert groundedness("", ["任何内容"]) == 0.0


def test_evaluate_empty_case_list():
    report = evaluate(None, [])  # type: ignore[arg-type]
    assert report.n == 0


def test_pipeline_meets_retrieval_baseline(pipeline):
    report = evaluate(pipeline, CASES)
    assert report.n == len(CASES)
    # Regression floors, not aspirations: the current config scores
    # hit_rate=1.00 mrr=1.00 ctx_precision=0.875 groundedness=1.00 on this set,
    # so a change that degrades retrieval fails CI instead of shipping.
    assert report.hit_rate == 1.0, report.summary()
    assert report.mrr == 1.0, report.summary()
    assert report.context_precision >= 0.8, report.summary()
    assert report.groundedness >= 0.95, report.summary()
    assert report.refusal_accuracy == 1.0, report.summary()


FEE_QUERIES = [
    ("泓远沪深300增强ETF 的托管费年费率是多少", "fund-hs300-etf", "0.10%"),
    ("泓远沪深300增强ETF 的申购费率怎么算", "fund-hs300-etf", "0.80%"),
    ("泓远稳健纯债债券型基金的管理费年费率", "fund-bond-stable", "0.30%"),
]


def _pipeline_with(retriever_cls, corpus_dir, settings):
    from fin_rag import MockLLM, RagPipeline
    from fin_rag.embeddings import build_embedder
    from fin_rag.ingest import documents_to_chunks, load_documents

    chunks = documents_to_chunks(load_documents(corpus_dir), settings)
    embedder = build_embedder(settings.embedding_provider, dim=settings.embedding_dim)
    retriever = retriever_cls(embedder, rrf_k=settings.retrieval.rrf_k)
    retriever.index(chunks)
    return RagPipeline(retriever=retriever, llm=MockLLM(), settings=settings)


def test_contextual_header_puts_the_right_section_first(corpus_dir, settings):
    """The header must make cross-section queries land on the answer section."""
    from fin_rag.retrieval import HybridRetriever

    pipeline = _pipeline_with(HybridRetriever, corpus_dir, settings)
    for question, doc, number in FEE_QUERIES:
        result = pipeline.answer(question)
        assert result.citations, question
        top = result.citations[0]
        assert top["source"].endswith("§费用结构"), (question, top["source"])
        assert _doc_root(str(top["doc_id"])) == doc, (question, top["doc_id"])
        assert number in result.answer, (question, result.answer[:80])


def test_without_header_fee_queries_hit_the_wrong_section_or_fund(corpus_dir, settings):
    """Ablation that keeps the README claim honest.

    Indexing bare chunk bodies breaks 2 of the 3 fee queries above. One of them
    returns the *other fund's* fee section — in a finance product that is worse
    than returning nothing, because the answer looks confident and is wrong.
    """
    from fin_rag.chunking import Chunk
    from fin_rag.retrieval import HybridRetriever

    class NoHeaderRetriever(HybridRetriever):
        def index(self, chunks):  # type: ignore[override]
            super().index(
                [
                    Chunk(doc_id=c.doc_id, chunk_id=c.chunk_id, text=c.text, context=c.context)
                    for c in chunks
                ]
            )

    pipeline = _pipeline_with(NoHeaderRetriever, corpus_dir, settings)
    failures = 0
    for question, doc, number in FEE_QUERIES:
        result = pipeline.answer(question)
        top = result.citations[0] if result.citations else {"doc_id": ""}
        wrong_doc = _doc_root(str(top["doc_id"])) != doc
        missing_number = number not in result.answer
        if wrong_doc or missing_number:
            failures += 1
    assert failures >= 2, "ablation must be measurably worse than the header variant"


def test_report_summary_is_printable(pipeline):
    report = evaluate(pipeline, CASES[:2])
    assert "hit_rate=" in report.summary()
    assert len(report.per_case) == 2
