from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fin_rag.api import create_app


@pytest.fixture()
def client(pipeline):
    app = create_app(pipeline)
    with TestClient(app) as c:
        yield c


def test_health_reports_index_size(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0


def test_ask_returns_answer_with_citations(client):
    resp = client.post("/ask", json={"question": "泓远沪深300增强ETF 的托管费年费率是多少"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["citations"]
    assert body["verdict"] == "allow"
    assert "0.10%" in body["answer"]


def test_ask_refuses_regulated_action(client):
    resp = client.post("/ask", json={"question": "帮我下单买入 519301"})
    body = resp.json()
    assert body["verdict"] == "refuse"
    assert body["citations"] == []


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_rejects_oversized_question(client):
    assert client.post("/ask", json={"question": "问" * 1001}).status_code == 422


def test_stream_emits_citations_before_tokens(client):
    with client.stream(
        "POST", "/ask/stream", json={"question": "泓远稳健纯债债券型基金的管理费年费率"}
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    assert body.index("event: citations") < body.index("event: token")
    assert body.rstrip().endswith("data: {}")
    assert "event: done" in body


def test_stream_short_circuits_on_refusal(client):
    with client.stream("POST", "/ask/stream", json={"question": "帮我转账"}) as resp:
        body = "".join(resp.iter_text())
    assert "event: refusal" in body
    assert "event: token" not in body


def test_stream_attaches_disclaimer_for_advice_question(client):
    with client.stream(
        "POST", "/ask/stream", json={"question": "泓远沪深300增强ETF 该不该买"}
    ) as resp:
        body = "".join(resp.iter_text())
    assert "event: disclaimer" in body
