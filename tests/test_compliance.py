from __future__ import annotations

import pytest

from fin_rag.compliance import Verdict, check_query, sanitise_answer


@pytest.mark.parametrize(
    "question",
    [
        "帮我下单买入 519301",
        "帮我转账到这个账户",
        "有没有内幕消息",
        "教我操纵股价",
    ],
)
def test_regulated_actions_are_refused(question):
    result = check_query(question)
    assert result.verdict is Verdict.REFUSE
    assert result.blocked


@pytest.mark.parametrize(
    "question",
    [
        "这只基金我该不该买",
        "519301 值不值得加仓",
        "推荐几个能买的基金",
        "Should I buy this fund?",
    ],
)
def test_advice_solicitation_gets_disclaimer_not_refusal(question):
    result = check_query(question)
    assert result.verdict is Verdict.ALLOW_WITH_DISCLAIMER
    assert "personal-investment-advice" in result.reasons
    assert result.disclaimer
    assert not result.blocked


@pytest.mark.parametrize(
    "question",
    ["明天会涨吗", "预测一下未来收益率", "有没有稳赚的产品"],
)
def test_prediction_requests_get_disclaimer(question):
    result = check_query(question)
    assert result.verdict is Verdict.ALLOW_WITH_DISCLAIMER
    assert any("prediction" in r or "advice" in r for r in result.reasons)


@pytest.mark.parametrize(
    "question",
    [
        "泓远沪深300增强ETF 的管理费率是多少",
        "这只基金 2024 年的最大回撤是多少",
        "投资者风险承受能力怎么分级",
    ],
)
def test_plain_information_queries_pass(question):
    result = check_query(question)
    assert result.verdict is Verdict.ALLOW
    assert result.reasons == []


def test_action_check_precedes_advice_check():
    # contains both an advice pattern and a regulated action; refusal must win
    result = check_query("这只基金该不该买，帮我下单")
    assert result.verdict is Verdict.REFUSE


def test_sanitise_strips_guaranteed_return_language():
    cleaned, notes = sanitise_answer("该产品保证收益，稳赚不赔。")
    assert "保证收益" not in cleaned
    assert "稳赚不赔" not in cleaned
    assert notes


def test_sanitise_rewrites_recommendation_framing():
    cleaned, notes = sanitise_answer("建议买入该基金。")
    assert "建议买入" not in cleaned
    assert "相关文档提到" in cleaned
    assert notes


def test_sanitise_leaves_clean_text_untouched():
    text = "该基金 2024 年净值增长率为 12.47%。"
    cleaned, notes = sanitise_answer(text)
    assert cleaned == text
    assert notes == []
