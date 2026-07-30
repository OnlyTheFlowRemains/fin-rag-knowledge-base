"""Compliance guardrails for financial Q&A.

This is the domain-specific layer, and the reason a finance RAG differs from a
generic one: in HK/mainland retail finance, an assistant that answers "should I
buy this fund" is a regulatory problem, not a product feature. Rules here are
deliberately conservative and auditable — every decision returns a reason
string that lands in the response payload.

Not legal advice; the rule set is illustrative and must be reviewed by a
compliance team before production use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    ALLOW = "allow"
    ALLOW_WITH_DISCLAIMER = "allow_with_disclaimer"
    REFUSE = "refuse"


@dataclass
class ComplianceResult:
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    disclaimer: str = ""

    @property
    def blocked(self) -> bool:
        return self.verdict is Verdict.REFUSE


DISCLAIMER = (
    "以上内容基于知识库文档整理，仅供信息参考，不构成投资建议。"
    "投资涉及风险，过往表现不代表未来回报，请自行判断或咨询持牌顾问。"
)

# Direct solicitation of a personal investment recommendation.
_ADVICE_PATTERNS = [
    re.compile(r"(该不该|应不应该|要不要|值不值得|能不能|可不可以)\s*(买|卖|加仓|减仓|清仓|抄底|梭哈)"),
    re.compile(r"(帮我|替我|给我)\s*(买|卖|下单|建仓|操作)"),
    re.compile(r"(推荐|给个).{0,6}(股票|基金|标的|代码)"),
    re.compile(r"should i (buy|sell|invest)", re.IGNORECASE),
]

# Asking the model to predict a specific future price/return.
_PREDICTION_PATTERNS = [
    re.compile(r"(明天|下周|下个月|明年|未来).{0,8}(涨|跌|会到|能到|走势|价格)"),
    re.compile(r"(预测|保证|一定).{0,6}(收益|回报|涨幅|收益率)"),
    re.compile(r"(稳赚|包赚|无风险高收益|必涨)"),
]

# Regulated actions a read-only knowledge assistant must never take.
_ACTION_PATTERNS = [
    re.compile(r"(下单|委托交易|转账|划转|开户|代客理财)"),
    re.compile(r"(内幕|老鼠仓|操纵股价|避税方案|洗钱)"),
]


def check_query(question: str) -> ComplianceResult:
    reasons: list[str] = []
    for pattern in _ACTION_PATTERNS:
        if pattern.search(question):
            reasons.append(f"regulated-action:{pattern.pattern}")
    if reasons:
        return ComplianceResult(Verdict.REFUSE, reasons)

    for pattern in _ADVICE_PATTERNS:
        if pattern.search(question):
            reasons.append("personal-investment-advice")
            break
    for pattern in _PREDICTION_PATTERNS:
        if pattern.search(question):
            reasons.append("price-prediction-or-guaranteed-return")
            break
    if reasons:
        # Not refused outright: we still answer with sourced facts, but strip
        # the recommendation framing and attach the disclaimer.
        return ComplianceResult(Verdict.ALLOW_WITH_DISCLAIMER, reasons, DISCLAIMER)
    return ComplianceResult(Verdict.ALLOW, [], DISCLAIMER)


_GUARANTEE_TERMS = [
    (re.compile(r"保证收益|稳赚不赔|无风险|必涨|一定能涨"), "[已移除绝对化收益表述]"),
    (re.compile(r"建议(买入|卖出|满仓|清仓)"), "相关文档提到"),
]


def sanitise_answer(answer: str) -> tuple[str, list[str]]:
    """Strip absolute-return / recommendation language from generated text.

    Belt-and-braces: the prompt already forbids it, but prompts are not a
    control. A regex post-filter is.
    """
    notes: list[str] = []
    cleaned = answer
    for pattern, replacement in _GUARANTEE_TERMS:
        cleaned, count = pattern.subn(replacement, cleaned)
        if count:
            notes.append(f"sanitised:{pattern.pattern}")
    return cleaned, notes


REFUSAL_MESSAGE = (
    "抱歉，该请求涉及交易操作或违规内容，本助手仅提供知识库文档的信息查询，无法处理。"
)
