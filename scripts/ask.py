"""CLI: index the corpus and answer a question.

    python scripts/ask.py "泓远沪深300增强ETF 的管理费年费率是多少"
    python scripts/ask.py --json "投资者风险承受能力如何分级"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fin_rag import RagPipeline, build_llm, load_settings  # noqa: E402
from fin_rag.ingest import build_from_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the financial knowledge base.")
    parser.add_argument("question", help="natural-language question")
    parser.add_argument("--json", action="store_true", help="emit the full JSON payload")
    parser.add_argument("--corpus", default=None, help="override corpus directory")
    args = parser.parse_args()

    settings = load_settings()
    retriever, chunks = build_from_corpus(args.corpus or settings.corpus_dir, settings)
    llm = build_llm(settings.llm_provider, model=settings.llm_model, base_url=settings.llm_base_url)
    pipeline = RagPipeline(retriever=retriever, llm=llm, settings=settings)

    result = pipeline.answer(args.question)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"indexed {len(chunks)} chunks from {settings.corpus_dir}")
    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")
    if result.citations:
        print("Sources:")
        for citation in result.citations:
            print(f"  [{citation['index']}] {citation['source']}  (score={citation['score']})")
    if result.disclaimer:
        print(f"\n{result.disclaimer}")
    print(f"\ntrace: {' -> '.join(result.trace)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
