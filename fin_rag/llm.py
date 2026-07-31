"""LLM abstraction.

The whole pipeline talks to the `LLM` protocol, never to a vendor SDK. That
buys two things: the test suite runs offline and deterministically, and
swapping Qwen for DeepSeek/OpenAI is a config change, not a refactor.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLM(Protocol):
    def complete(self, prompt: str, *, system: str | None = None) -> str: ...

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]: ...


class MockLLM:
    """Deterministic stand-in used by tests, CI and the offline demo.

    It is not a toy: it extracts sentences from the retrieved context that
    share tokens with the question. That means the RAG plumbing (retrieval,
    fusion, citation, refusal) is genuinely exercised end-to-end without a
    network call or an API key.
    """

    def __init__(self, *, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences
        self.calls: list[str] = []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        question = _extract_block(prompt, "QUESTION")
        context = _extract_block(prompt, "CONTEXT")
        if not context.strip():
            return "根据现有资料无法回答该问题。"
        keywords = _tokens(question)
        scored: list[tuple[float, str]] = []
        # Respect the [1] [2] ... block order. A real model attends to the
        # ranking it is given; without this the mock happily quotes a
        # lower-ranked block about a *different* product, which in a finance
        # answer is a correctness bug, not cosmetic.
        for block_idx, block in enumerate(_split_blocks(context)):
            rank_weight = 1.0 / (1.0 + 0.35 * block_idx)
            for sentence in _split_sentences(block):
                hits = sum(1 for kw in keywords if kw in sentence)
                if hits:
                    scored.append((hits * rank_weight, sentence))
        if not scored:
            return "根据现有资料无法回答该问题。"
        scored.sort(key=lambda pair: -pair[0])
        picked = [s for _, s in scored[: self.max_sentences]]
        return " ".join(picked)

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        text = self.complete(prompt, system=system)
        for i in range(0, len(text), 12):
            yield text[i : i + 12]


class OpenAICompatibleLLM:
    """Works with any OpenAI-compatible endpoint (Qwen/DashScope, DeepSeek,
    Moonshot, vLLM, Ollama). Imported lazily so `openai` stays optional."""

    def __init__(self, *, model: str, base_url: str, api_key: str | None = None) -> None:
        import os

        from openai import OpenAI  # local import: optional dependency

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key or os.getenv("LLM_API_KEY", ""))

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=self._messages(prompt, system), temperature=0.1
        )
        return resp.choices[0].message.content or ""

    def stream(self, prompt: str, *, system: str | None = None) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt, system),
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def build_llm(provider: str, *, model: str, base_url: str) -> LLM:
    if provider == "mock":
        return MockLLM()
    if provider in {"openai", "qwen", "deepseek", "compatible"}:
        return OpenAICompatibleLLM(model=model, base_url=base_url)
    raise ValueError(f"unknown LLM provider: {provider!r}")


_CJK_STOP = {
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "对",
    "有",
    "为",
    "及",
    "请",
    "问",
    "吗",
    "如何",
    "什么",
}


def _tokens(text: str) -> list[str]:
    """Cheap bilingual tokenizer: ASCII words + CJK bigrams."""
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", text) if len(w) > 1]
    cjk = re.findall(r"[一-鿿]+", text)
    grams: list[str] = []
    for run in cjk:
        if len(run) == 1:
            grams.append(run)
            continue
        grams.extend(run[i : i + 2] for i in range(len(run) - 1))
    return [t for t in words + grams if t not in _CJK_STOP]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；\n])|(?<=[.!?])\s", text)
    return [p.strip() for p in parts if p and p.strip()]


def _split_blocks(context: str) -> list[str]:
    """Split a numbered context ("[1] ... [2] ...") back into ranked blocks."""
    pieces = re.split(r"\n\n(?=\[\d+\]\s)", context.strip())
    return [p for p in pieces if p.strip()]


def _extract_block(prompt: str, label: str) -> str:
    match = re.search(rf"<{label}>(.*?)</{label}>", prompt, re.DOTALL)
    return match.group(1) if match else ""


def concat_tokens(chunks: Sequence[str]) -> str:
    return "".join(chunks)
