"""
Two-phase clinical research agent.

Phase 1 — Mandatory tool execution (no LLM routing decision):
    GraphRAG is called directly and always runs before the LLM sees the query.

Phase 2 — Evidence-grounded synthesis:
    The LLM receives only the retrieved evidence + query. A strict system prompt
    prevents parametric memory use. If GraphRAG returns found=false, the LLM
    responds with a fixed "no evidence" message — no speculation.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import httpx
from openai import APIConnectionError, APITimeoutError
from langchain_core.messages import HumanMessage, SystemMessage

from .config import AgentConfig, load_config
from .llm_factory import build_llm, check_llm_health
from .tools.graphrag import GraphRAGTool

logger = logging.getLogger(__name__)

_NO_EVIDENCE_RESPONSE = "No evidence found for this query."
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_THINK_START = "<think>"
_THINK_END = "</think>"


class LLMUnavailableError(RuntimeError):
    """Raised when neither the configured primary nor fallback LLM can serve."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class SynthesisResult:
    """A grounded synthesis together with the serving model metadata."""

    text: str
    model: str | None
    fallback_used: bool


@dataclass(frozen=True)
class SynthesisStreamEvent:
    """One token or serving-model change emitted during streamed synthesis."""

    type: Literal["token", "meta"]
    text: str = ""
    model: str | None = None
    fallback_used: bool = False


@dataclass(frozen=True)
class PreparedSynthesisStream:
    """A preflighted synthesis stream ready for HTTP transport."""

    model: str
    fallback_used: bool
    events: Iterator[SynthesisStreamEvent]


@dataclass
class RunResult:
    query: str
    evidence: dict[str, Any]
    synthesis: str
    latency_ms: float
    synthesis_model: str | None = None
    fallback_used: bool = False
    found: bool = field(init=False)

    def __post_init__(self) -> None:
        self.found = bool(self.evidence.get("found", False))


def _format_evidence(evidence: dict[str, Any]) -> str:
    """Render GraphRAG output as a readable evidence block for the LLM."""
    if not evidence.get("found", False):
        return "No evidence retrieved."

    parts: list[str] = []

    vector_results: list[dict] = evidence.get("vector_results", [])
    for i, hit in enumerate(vector_results, 1):
        source = hit.get("source", "unknown")
        score = hit.get("score", 0)
        content = hit.get("content", "").strip()
        parts.append(f"[{i}] source={source} score={score:.4f}\n{content}")

    graph_facts: list[str] = evidence.get("graph_facts", [])
    if graph_facts:
        parts.append("\nKnowledge graph facts:")
        parts.extend(f"  • {fact}" for fact in graph_facts)

    return "\n\n".join(parts) if parts else "No evidence retrieved."


def _strip_reasoning_block(text: str) -> str:
    """Remove a leading local-model reasoning block from a completed response."""
    return _THINK_BLOCK_RE.sub("", text, count=1)


def _visible_stream_tokens(chunks: Iterator[Any]) -> Iterator[str]:
    """Yield answer tokens while suppressing a leading `<think>` block."""
    state: Literal["prefix", "thinking", "answer"] = "prefix"
    buffer = ""
    trim_answer_prefix = False

    for chunk in chunks:
        content = chunk.content or ""
        if not content:
            continue
        token = str(content)

        if state == "answer":
            if trim_answer_prefix:
                token = token.lstrip("\r\n")
                if not token:
                    continue
                trim_answer_prefix = False
            yield token
            continue

        buffer += token
        if state == "prefix":
            candidate = buffer.lstrip()
            if _THINK_START.startswith(candidate) and len(candidate) < len(_THINK_START):
                continue
            if candidate.startswith(_THINK_START):
                state = "thinking"
                buffer = candidate[len(_THINK_START):]
            else:
                state = "answer"
                yield buffer
                buffer = ""
                continue

        end_index = buffer.find(_THINK_END)
        if end_index >= 0:
            state = "answer"
            answer_prefix = buffer[end_index + len(_THINK_END):].lstrip("\r\n")
            buffer = ""
            if answer_prefix:
                trim_answer_prefix = False
                yield answer_prefix
            else:
                trim_answer_prefix = True
        else:
            buffer = buffer[-(len(_THINK_END) - 1):]

    if state == "prefix" and buffer:
        yield buffer
    elif state == "thinking":
        raise LLMUnavailableError(
            "The synthesis model returned an unterminated reasoning block."
        )


class Agent:
    """Deterministic two-phase pipeline: GraphRAG retrieval → grounded synthesis."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        params = config.model_params.model_dump(exclude_none=True)
        self.llm = build_llm(config.model, **params)
        self.fallback_llm = (
            build_llm(config.fallback_model, **params)
            if config.fallback_model
            else None
        )
        self.graphrag = GraphRAGTool(config.graphrag.model_dump())

    @classmethod
    def from_config(cls, path: Path | None = None) -> "Agent":
        """Construct an Agent from the app.yaml agentic_reasoning section."""
        return cls(load_config(path))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, query: str, evidence: dict[str, Any]) -> list:
        context = _format_evidence(evidence)
        user_content = (
            f"[QUERY]\n{query}\n\n"
            f"[EVIDENCE]\n{context}\n[/EVIDENCE]"
        )
        return [
            SystemMessage(content=self.config.system_prompt),
            HumanMessage(content=user_content),
        ]

    def _select_synthesis_llm(self) -> tuple[Any, str, bool]:
        """Return a healthy primary model or an explicitly configured fallback."""
        timeout = self.config.health_check_timeout_seconds
        primary_health = check_llm_health(self.config.model, timeout)
        if primary_health.available:
            return self.llm, self.config.model, False

        fallback_model = self.config.fallback_model
        if fallback_model and self.fallback_llm is not None:
            fallback_health = check_llm_health(fallback_model, timeout)
            if fallback_health.available:
                logger.warning(
                    "Primary LLM unavailable; using configured fallback: primary=%s fallback=%s",
                    self.config.model,
                    fallback_model,
                )
                return self.fallback_llm, fallback_model, True
            raise LLMUnavailableError(
                "Synthesis is unavailable: "
                f"primary ({self.config.model}) health check failed: {primary_health.detail}; "
                f"fallback ({fallback_model}) health check failed: {fallback_health.detail}"
            )

        raise LLMUnavailableError(
            "Synthesis is unavailable: "
            f"primary ({self.config.model}) health check failed: {primary_health.detail}; "
            "no fallback model is configured."
        )

    def synthesize(self, query: str, evidence: dict[str, Any]) -> SynthesisResult:
        """Produce a strictly evidence-grounded synthesis with explicit failover."""
        if not evidence.get("found", False):
            return SynthesisResult(
                text=_NO_EVIDENCE_RESPONSE,
                model=None,
                fallback_used=False,
            )

        messages = self._build_messages(query, evidence)
        llm, model, fallback_used = self._select_synthesis_llm()
        try:
            response = llm.invoke(messages)
        except (APIConnectionError, APITimeoutError, httpx.HTTPError, ConnectionError, TimeoutError) as exc:
            if fallback_used or not self.config.fallback_model or self.fallback_llm is None:
                raise LLMUnavailableError(
                    f"Synthesis invocation failed for {model}: {type(exc).__name__}: {exc}"
                ) from exc

            fallback_model = self.config.fallback_model
            fallback_health = check_llm_health(
                fallback_model,
                self.config.health_check_timeout_seconds,
            )
            if not fallback_health.available:
                raise LLMUnavailableError(
                    "Synthesis is unavailable after primary invocation failed: "
                    f"primary ({model}) error={type(exc).__name__}: {exc}; "
                    f"fallback ({fallback_model}) health check failed: {fallback_health.detail}"
                ) from exc

            logger.warning(
                "Primary LLM invocation failed; retrying configured fallback: "
                "primary=%s fallback=%s error=%s",
                model,
                fallback_model,
                type(exc).__name__,
            )
            try:
                response = self.fallback_llm.invoke(messages)
            except (APIConnectionError, APITimeoutError, httpx.HTTPError, ConnectionError, TimeoutError) as fallback_exc:
                raise LLMUnavailableError(
                    f"Synthesis invocation failed for fallback {fallback_model}: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc
            fallback_text = _strip_reasoning_block(str(response.content or "")).strip()
            return SynthesisResult(
                text=fallback_text or _NO_EVIDENCE_RESPONSE,
                model=fallback_model,
                fallback_used=True,
            )

        response_text = _strip_reasoning_block(str(response.content or "")).strip()
        return SynthesisResult(
            text=response_text or _NO_EVIDENCE_RESPONSE,
            model=model,
            fallback_used=fallback_used,
        )

    def _stream_synthesis_events(
        self,
        messages: list[Any],
        llm: Any,
        model: str,
        fallback_used: bool,
    ) -> Iterator[SynthesisStreamEvent]:
        emitted_token = False
        try:
            for token in _visible_stream_tokens(llm.stream(messages)):
                emitted_token = True
                yield SynthesisStreamEvent(type="token", text=token)
            return
        except (
            APIConnectionError,
            APITimeoutError,
            httpx.HTTPError,
            ConnectionError,
            TimeoutError,
        ) as exc:
            fallback_model = self.config.fallback_model
            if (
                emitted_token
                or fallback_used
                or not fallback_model
                or self.fallback_llm is None
            ):
                raise LLMUnavailableError(
                    f"Synthesis stream failed for {model}: {type(exc).__name__}: {exc}"
                ) from exc

            fallback_health = check_llm_health(
                fallback_model,
                self.config.health_check_timeout_seconds,
            )
            if not fallback_health.available:
                raise LLMUnavailableError(
                    "Synthesis stream is unavailable after primary transport failure: "
                    f"primary ({model}) error={type(exc).__name__}: {exc}; "
                    f"fallback ({fallback_model}) health check failed: {fallback_health.detail}"
                ) from exc

            logger.warning(
                "Primary synthesis stream failed before output; switching fallback: "
                "primary=%s fallback=%s error=%s",
                model,
                fallback_model,
                type(exc).__name__,
            )
            yield SynthesisStreamEvent(
                type="meta",
                model=fallback_model,
                fallback_used=True,
            )

            try:
                for token in _visible_stream_tokens(self.fallback_llm.stream(messages)):
                    yield SynthesisStreamEvent(type="token", text=token)
            except (
                APIConnectionError,
                APITimeoutError,
                httpx.HTTPError,
                ConnectionError,
                TimeoutError,
            ) as fallback_exc:
                raise LLMUnavailableError(
                    f"Synthesis stream failed for fallback {fallback_model}: "
                    f"{type(fallback_exc).__name__}: {fallback_exc}"
                ) from fallback_exc

    def prepare_synthesis_stream(
        self,
        query: str,
        evidence: dict[str, Any],
    ) -> PreparedSynthesisStream:
        """Preflight a grounded stream without repeating evidence retrieval."""
        if not evidence.get("found", False):
            raise ValueError("Cannot stream synthesis without retrieved evidence.")

        messages = self._build_messages(query, evidence)
        llm, model, fallback_used = self._select_synthesis_llm()
        return PreparedSynthesisStream(
            model=model,
            fallback_used=fallback_used,
            events=self._stream_synthesis_events(
                messages,
                llm,
                model,
                fallback_used,
            ),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> RunResult:
        """Blocking two-phase run. Returns a RunResult with synthesis and evidence."""
        t0 = time.perf_counter()

        # Phase 1: deterministic tool execution
        logger.info("Phase 1 — GraphRAG retrieval for query: %s", query)
        evidence = self.graphrag.cached_execute(query)
        if not isinstance(evidence, dict):
            evidence = {"found": False, "error": str(evidence)}
        logger.info(
            "Phase 1 complete — found=%s, vector_hits=%d, graph_facts=%d",
            evidence.get("found"),
            len(evidence.get("vector_results", [])),
            len(evidence.get("graph_facts", [])),
        )

        # Phase 2: grounded synthesis
        logger.info("Phase 2 — LLM synthesis from evidence")
        synthesis_result = self.synthesize(query, evidence)

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("Run complete in %.0fms", latency_ms)

        return RunResult(
            query=query,
            evidence=evidence,
            synthesis=synthesis_result.text,
            latency_ms=latency_ms,
            synthesis_model=synthesis_result.model,
            fallback_used=synthesis_result.fallback_used,
        )

    def stream(self, query: str) -> Iterator[str]:
        """Two-phase streaming run. Phase 1 blocks; Phase 2 streams synthesis tokens.

        Yields string chunks as they arrive. Callers can accumulate them to reconstruct
        the full synthesis. Evidence is retrievable via agent.last_evidence after the
        generator is exhausted.
        """
        # Phase 1: blocking (must complete before LLM sees anything)
        logger.info("Phase 1 — GraphRAG retrieval (stream mode)")
        evidence = self.graphrag.cached_execute(query)
        if not isinstance(evidence, dict):
            evidence = {"found": False, "error": str(evidence)}
        self.last_evidence = evidence

        logger.info(
            "Phase 1 complete — found=%s, vector_hits=%d",
            evidence.get("found"),
            len(evidence.get("vector_results", [])),
        )

        if not evidence.get("found", False):
            yield _NO_EVIDENCE_RESPONSE
            return

        logger.info("Phase 2 — streaming synthesis")
        prepared = self.prepare_synthesis_stream(query, evidence)
        for event in prepared.events:
            if event.type == "token":
                yield event.text

    def run_json(self, query: str) -> dict[str, Any]:
        """Run and return a JSON-serialisable dict (for server/CLI use)."""
        result = self.run(query)
        return {
            "query": result.query,
            "synthesis": result.synthesis,
            "found": result.found,
            "synthesis_model": result.synthesis_model,
            "fallback_used": result.fallback_used,
            "latency_ms": round(result.latency_ms, 1),
            "evidence": result.evidence,
        }
