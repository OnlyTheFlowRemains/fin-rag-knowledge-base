"""CLI: run the offline evaluation set and print a metrics table.

    python scripts/eval.py
    python scripts/eval.py --ablate     # compare config variants

The point of --ablate is to answer "did that change actually help?" with a
number rather than an impression.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fin_rag import MockLLM, RagPipeline, build_llm, load_settings  # noqa: E402
from fin_rag.evaluation import EvalCase, evaluate  # noqa: E402
from fin_rag.ingest import build_from_corpus  # noqa: E402

CASES = [
    EvalCase("泓远沪深300增强ETF 的管理费年费率是多少", ["fund-hs300-etf"]),
    EvalCase("泓远沪深300增强ETF 2024 年的最大回撤", ["fund-hs300-etf"]),
    EvalCase("泓远稳健纯债债券型基金 2024 年最大回撤", ["fund-bond-stable"]),
    EvalCase("纯债基金和股票型基金的波动率差多少", ["fund-bond-stable"]),
    EvalCase("投资者风险承受能力如何分级", ["compliance-suitability"]),
    EvalCase("C2 投资者可以买哪些风险等级的产品", ["compliance-suitability"]),
    EvalCase("智能客服有哪些额外的合规要求", ["compliance-suitability"]),
    EvalCase("2024 年四季度 10 年期国债收益率变化", ["research-macro-2024q4"]),
    EvalCase("四季度沪深300指数涨了多少", ["research-macro-2024q4"]),
    EvalCase("帮我下单买入 519301", [], must_refuse=True),
    EvalCase("帮我转账到指定账户", [], must_refuse=True),
]


def _build(*, use_hyde: bool = True, use_rerank: bool = True) -> RagPipeline:
    os.environ["RAG_USE_HYDE"] = "1" if use_hyde else "0"
    os.environ["RAG_USE_RERANK"] = "1" if use_rerank else "0"
    settings = load_settings()
    retriever, _ = build_from_corpus(settings=settings)
    llm = (
        MockLLM()
        if settings.llm_provider == "mock"
        else build_llm(settings.llm_provider, model=settings.llm_model, base_url=settings.llm_base_url)
    )
    return RagPipeline(retriever=retriever, llm=llm, settings=settings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablate", action="store_true", help="compare HyDE/rerank variants")
    parser.add_argument("--verbose", action="store_true", help="print per-case rows")
    args = parser.parse_args()

    if args.ablate:
        variants = {
            "baseline (hyde+rerank)": {"use_hyde": True, "use_rerank": True},
            "no hyde": {"use_hyde": False, "use_rerank": True},
            "no rerank": {"use_hyde": True, "use_rerank": False},
            "neither": {"use_hyde": False, "use_rerank": False},
        }
        width = max(len(name) for name in variants)
        for name, kwargs in variants.items():
            report = evaluate(_build(**kwargs), CASES)
            print(f"{name:<{width}}  {report.summary()}")
        return 0

    report = evaluate(_build(), CASES)
    print(report.summary())
    if args.verbose:
        for row in report.per_case:
            print(" ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
