"""Offline tests for local-model endpoint readiness checks."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.llm_factory import LLMHealth, check_llm_health


def _check_with_payload(model: str, payload: object) -> tuple[LLMHealth, MagicMock]:
    client = MagicMock()
    client.__enter__.return_value = client
    response = MagicMock()
    response.json.return_value = payload
    client.get.return_value = response
    with patch("src.llm_factory.httpx.Client", return_value=client):
        health = check_llm_health(model)
    return health, client


@pytest.mark.parametrize(
    ("model", "payload", "expected_url"),
    [
        (
            "lmstudio/request-alias",
            {"data": [{"id": "loaded-model"}]},
            "http://localhost:1234/v1/models",
        ),
        (
            "sglang/Qwen/model",
            {"data": [{"id": "Qwen/model"}]},
            "http://localhost:30000/v1/models",
        ),
        (
            "ollama/qwen:latest",
            {"models": [{"name": "qwen:latest"}]},
            "http://localhost:11434/api/tags",
        ),
        (
            "ollama/qwen:latest",
            {"models": [{"model": "qwen:latest"}]},
            "http://localhost:11434/api/tags",
        ),
    ],
)
def test_health_accepts_usable_model_list(
    model: str, payload: object, expected_url: str
) -> None:
    health, client = _check_with_payload(model, payload)

    assert isinstance(health, LLMHealth)
    assert health.available is True
    assert health.detail is None
    client.get.assert_called_once_with(expected_url)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        ("lmstudio/model", {"data": []}),
        ("sglang/model", {"data": []}),
        ("ollama/model", {"models": []}),
    ],
)
def test_health_rejects_empty_model_list(model: str, payload: object) -> None:
    health, _ = _check_with_payload(model, payload)

    assert health.available is False
    assert health.detail is not None
    assert "No usable models" in health.detail
    assert "retry" in health.detail


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        ("lmstudio/model", {}),
        ("lmstudio/model", {"data": "not-a-list"}),
        ("lmstudio/model", {"data": [{}]}),
        ("sglang/model", {"data": [{"id": "  "}]}),
        ("ollama/model", {"models": [{"name": "", "model": ""}]}),
        ("ollama/model", {"data": [{"id": "wrong-schema"}]}),
    ],
)
def test_health_rejects_malformed_model_list(model: str, payload: object) -> None:
    health, _ = _check_with_payload(model, payload)

    assert health.available is False
    assert health.detail is not None
    assert "Malformed model-list response" in health.detail


def test_health_rejects_invalid_json() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    response = MagicMock()
    response.json.side_effect = json.JSONDecodeError("bad JSON", "", 0)
    client.get.return_value = response

    with patch("src.llm_factory.httpx.Client", return_value=client):
        health = check_llm_health("lmstudio/model")

    assert health.available is False
    assert health.detail is not None
    assert "expected JSON" in health.detail


def test_health_preserves_transport_error_detail() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.side_effect = httpx.ConnectError("connection refused")

    with patch("src.llm_factory.httpx.Client", return_value=client):
        health = check_llm_health("lmstudio/model", timeout_seconds=0.25)

    assert health.available is False
    assert health.detail is not None
    assert health.detail.startswith("ConnectError: connection refused")
