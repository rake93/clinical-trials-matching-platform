"""FastAPI contract tests for synthesis availability and fallback metadata."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from src.agent import (
    LLMUnavailableError,
    PreparedSynthesisStream,
    SynthesisInputTooLargeError,
    SynthesisResult,
    SynthesisStreamEvent,
)
from src import server
from src.tools.graphrag import GraphRAGUnavailableError


EVIDENCE: dict[str, Any] = {
    "found": True,
    "vector_results": [
        {
            "score": 0.82,
            "reranker_score": 0.01,
            "content": "Evidence content.",
            "source": "paper.pdf",
            "chunk_index": 2,
            "context": "Guideline",
            "page_number": 4,
            "char_start": 320,
            "char_end": 337,
        }
    ],
    "graph_facts": ["DRUG --[TREATS]--> CONDITION"],
    "graph_anchor": "DRUG",
}


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.graphrag.cached_execute.return_value = EVIDENCE
    return agent


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    mock_agent: MagicMock,
) -> TestClient:

    async def get_agent() -> MagicMock:
        return mock_agent

    monkeypatch.setattr(server, "_get_agent", get_agent)
    server._evidence_cache.clear()
    return TestClient(server.app)


def test_match_uses_vector_relevance_not_raw_reranker_score(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    response = client.post("/api/match", data={"query": "clinical query"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["matches"][0]["score"] == 0.82
    assert payload["matches"][0]["rankScore"] == 0.01
    assert payload["matches"][0]["pageNumber"] == 4
    assert payload["matches"][0]["charStart"] == 320
    assert payload["matches"][0]["charEnd"] == 337
    assert payload["graphAnchor"] == "DRUG"
    assert isinstance(payload["evidenceId"], str)
    mock_agent.graphrag.cached_execute.assert_called_once_with(
        {
            "query": "clinical query",
            "target": "literature",
            "source": None,
            "source_slug": None,
            "limit": 10,
        }
    )


def test_match_forwards_active_document_context(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    response = client.post(
        "/api/match",
        data={
            "query": "prescribed dosage",
            "target": "patient_context",
            "source": "Adobe Scan.pdf",
            "source_slug": "Adobe Scan",
            "top_k": "4",
        },
    )

    assert response.status_code == 200
    mock_agent.graphrag.cached_execute.assert_called_once_with(
        {
            "query": "prescribed dosage",
            "target": "patient_context",
            "source": "Adobe Scan.pdf",
            "source_slug": "Adobe Scan",
            "limit": 4,
        }
    )


def test_match_requires_source_for_patient_context(client: TestClient) -> None:
    response = client.post(
        "/api/match",
        data={"query": "prescribed dosage", "target": "patient_context"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "source_required"


def test_match_returns_structured_empty_outcome(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    mock_agent.graphrag.cached_execute.return_value = {
        "found": False,
        "vector_results": [],
        "graph_facts": [],
        "empty": {
            "code": "source_mismatch",
            "message": "Wrong source.",
            "action": "Use the active source.",
        },
    }

    response = client.post(
        "/api/match",
        data={
            "query": "from [wrong.pdf]",
            "target": "patient_context",
            "source": "active.pdf",
            "source_slug": "active",
        },
    )

    assert response.status_code == 200
    assert response.json()["evidenceId"] is None
    assert response.json()["empty"]["code"] == "source_mismatch"


def test_match_surfaces_retrieval_outage(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    mock_agent.graphrag.cached_execute.side_effect = GraphRAGUnavailableError(
        "Qdrant unavailable"
    )

    response = client.post("/api/match", data={"query": "clinical query"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "retrieval_unavailable"


def test_synthesis_returns_fallback_metadata(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MagicMock()
    agent.graphrag.cached_execute.return_value = EVIDENCE
    agent.synthesize.return_value = SynthesisResult(
        text="Grounded fallback synthesis.",
        model="lmstudio/fallback-model",
        fallback_used=True,
    )

    async def get_agent() -> MagicMock:
        return agent

    monkeypatch.setattr(server, "_get_agent", get_agent)
    response = client.post(
        "/api/synthesis",
        json={"query": "clinical query", "evidence": []},
    )

    assert response.status_code == 200
    assert response.json() == {
        "synthesis": "Grounded fallback synthesis.",
        "model": "lmstudio/fallback-model",
        "fallbackUsed": True,
        "tokensUsed": None,
    }


def test_synthesis_returns_retryable_503_when_no_provider_is_available(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = MagicMock()
    agent.graphrag.cached_execute.return_value = EVIDENCE
    agent.synthesize.side_effect = LLMUnavailableError("primary and fallback unavailable")

    async def get_agent() -> MagicMock:
        return agent

    monkeypatch.setattr(server, "_get_agent", get_agent)
    response = client.post(
        "/api/synthesis",
        json={"query": "clinical query", "evidence": []},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "synthesis_unavailable",
        "message": "primary and fallback unavailable",
        "retryable": True,
    }


def test_synthesis_returns_typed_preflight_budget_error(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    match_response = client.post("/api/match", data={"query": "clinical query"})
    evidence_id = match_response.json()["evidenceId"]
    mock_agent.synthesize.side_effect = SynthesisInputTooLargeError(
        "Shorten the query and retry."
    )

    response = client.post(
        "/api/synthesis",
        json={"query": "clinical query", "evidenceId": evidence_id},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "synthesis_input_too_large",
        "message": "Shorten the query and retry.",
        "retryable": False,
    }


def test_synthesis_stream_emits_metadata_tokens_and_done(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    match_response = client.post("/api/match", data={"query": "clinical query"})
    evidence_id = match_response.json()["evidenceId"]
    mock_agent.prepare_synthesis_stream.return_value = PreparedSynthesisStream(
        model="lmstudio/test-model",
        fallback_used=False,
        events=iter(
            [
                SynthesisStreamEvent(type="token", text="Grounded"),
                SynthesisStreamEvent(type="token", text=" answer"),
            ]
        ),
    )

    response = client.post(
        "/api/synthesis/stream",
        json={"query": "clinical query", "evidenceId": evidence_id},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type":"meta"' in response.text
    assert '"text":"Grounded"' in response.text
    assert '"text":" answer"' in response.text
    assert response.text.rstrip().endswith('data: {"type":"done"}')
    mock_agent.prepare_synthesis_stream.assert_called_once()


def test_synthesis_stream_returns_typed_preflight_budget_error(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    match_response = client.post("/api/match", data={"query": "clinical query"})
    evidence_id = match_response.json()["evidenceId"]
    mock_agent.prepare_synthesis_stream.side_effect = SynthesisInputTooLargeError(
        "Shorten the query and retry."
    )

    response = client.post(
        "/api/synthesis/stream",
        json={"query": "clinical query", "evidenceId": evidence_id},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "code": "synthesis_input_too_large",
        "message": "Shorten the query and retry.",
        "retryable": False,
    }


def test_synthesis_stream_emits_error_without_done_after_provider_failure(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    match_response = client.post("/api/match", data={"query": "clinical query"})
    evidence_id = match_response.json()["evidenceId"]

    def failing_events():
        raise LLMUnavailableError("context overflow")
        yield

    mock_agent.prepare_synthesis_stream.return_value = PreparedSynthesisStream(
        model="lmstudio/test-model",
        fallback_used=True,
        events=failing_events(),
    )

    response = client.post(
        "/api/synthesis/stream",
        json={"query": "clinical query", "evidenceId": evidence_id},
    )

    assert response.status_code == 200
    assert '"type":"error"' in response.text
    assert '"code":"synthesis_unavailable"' in response.text
    assert '"type":"done"' not in response.text


def test_synthesis_stream_rejects_unknown_evidence(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    response = client.post(
        "/api/synthesis/stream",
        json={"query": "clinical query", "evidenceId": "expired"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "evidence_expired"
    mock_agent.prepare_synthesis_stream.assert_not_called()


def test_synthesis_stream_rejects_query_mismatch(
    client: TestClient,
    mock_agent: MagicMock,
) -> None:
    match_response = client.post("/api/match", data={"query": "clinical query"})
    evidence_id = match_response.json()["evidenceId"]

    response = client.post(
        "/api/synthesis/stream",
        json={"query": "different query", "evidenceId": evidence_id},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "evidence_mismatch"
