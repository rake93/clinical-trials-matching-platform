"""
Unit tests for the GraphRAG tool.

Qdrant, Neo4j, and SentenceTransformer are mocked so tests run offline.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.graphrag import (
    GraphRAGTool,
    GraphRAGUnavailableError,
    _extract_keywords,
)


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "qdrant_url": "http://localhost:6333",
    "collection": "test",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "model_cache_dir": "data/models",
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_username": "neo4j",
    "neo4j_password": "password",
    "scope": "literature",
    "min_relevance_score": 0.35,
    "limit": 2,
    "neo4j_limit": 5,
    "reranker_model": None,
    "cache_ttl": 60,
    "cache_maxsize": 32,
}

MULTI_TARGET_CONFIG = {
    **BASE_CONFIG,
    "retrieval_targets": [
        {
            "name": "patient_context",
            "collection": "patient_context",
            "scope": "patient_context",
        },
        {
            "name": "literature",
            "collection": "medical_papers",
            "scope": "literature",
        },
    ],
}


# ---------------------------------------------------------------------------
# _extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_filters_stop_words(self):
        keywords = _extract_keywords("what are the side effects of metformin")
        assert "what" not in keywords
        assert "are" not in keywords
        assert "the" not in keywords

    def test_extracts_meaningful_words(self):
        keywords = _extract_keywords("GLP-1 agonists in type 2 diabetes management")
        # Should contain substantive terms
        assert len(keywords) > 0

    def test_max_keywords_respected(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        keywords = _extract_keywords(text, max_keywords=3)
        assert len(keywords) <= 3

    def test_empty_query(self):
        assert _extract_keywords("") == []

    def test_short_words_filtered(self):
        keywords = _extract_keywords("is it ok to use it")
        # All words <= 2 chars should be excluded
        assert all(len(k) > 2 for k in keywords)


# ---------------------------------------------------------------------------
# GraphRAGTool.execute — mocked clients
# ---------------------------------------------------------------------------

class TestGraphRAGToolExecute:
    def _make_tool(self, config: dict | None = None) -> GraphRAGTool:
        return GraphRAGTool(config or BASE_CONFIG)

    def test_returns_found_true_on_vector_hits(self):
        tool = self._make_tool()
        mock_vector_results = [
            {
                "score": 0.8,
                "content": "Metformin reduces HbA1c.",
                "source": "doc.pdf",
                "chunk_id": "c1",
                "chunk_index": 0,
                "context": None,
            }
        ]
        with patch.object(tool, "_vector_search", return_value=mock_vector_results), \
             patch.object(tool, "_graph_context", return_value=[]):
            result = tool.execute("metformin diabetes")

        assert result["found"] is True
        assert len(result["vector_results"]) == 1
        assert result["vector_results"][0]["content"] == "Metformin reduces HbA1c."

    def test_returns_found_false_on_empty_results(self):
        tool = self._make_tool()
        with patch.object(tool, "_vector_search", return_value=[]), \
             patch.object(tool, "_graph_context", return_value=[]):
            result = tool.execute("completely unknown term xyz")

        assert result["found"] is False
        assert result["vector_results"] == []

    def test_rejects_low_relevance_vector_results(self):
        tool = self._make_tool()
        low_relevance = [
            {
                "score": 0.01,
                "content": "Unrelated chart text.",
                "source": "patient.pdf",
                "chunk_id": "c1",
                "chunk_index": 0,
                "context": None,
            }
        ]
        with patch.object(tool, "_vector_search", return_value=low_relevance), \
             patch.object(tool, "_graph_context", return_value=[]):
            result = tool.execute("post-exposure prophylaxis")

        assert result["found"] is False
        assert result["vector_results"] == []
        assert result["empty"]["code"] == "no_relevant_evidence"

    def test_graph_facts_included(self):
        tool = self._make_tool()
        mock_vector_results = [
            {"score": 0.7, "content": "Some content.", "source": "doc.pdf",
             "chunk_id": "c1", "chunk_index": 0, "context": None}
        ]
        graph_facts = ["DrugA --[TREATS]--> DiseaseB"]
        with patch.object(tool, "_vector_search", return_value=mock_vector_results), \
             patch.object(tool, "_graph_context", return_value=graph_facts):
            result = tool.execute("DrugA treatment")

        assert result["graph_facts"] == graph_facts
        assert result["graph_anchor"] == "DrugA"

    def test_empty_query_returns_error(self):
        tool = self._make_tool()
        result = tool.execute("")
        assert "Error" in str(result)

    def test_vector_search_failure_is_explicit(self):
        tool = self._make_tool()
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.query_points.side_effect = ConnectionError("Qdrant unavailable")

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.0] * 384

        tool._qdrant = mock_client
        tool._embedder = mock_embedder

        with pytest.raises(GraphRAGUnavailableError, match="Vector retrieval failed"):
            tool.execute("some query")

    def test_patient_context_requires_source(self):
        tool = self._make_tool(MULTI_TARGET_CONFIG)

        result = tool.execute(
            {"query": "prescribed dosage", "target": "patient_context"}
        )

        assert result == "Error: patient_context retrieval requires an exact source filename."

    def test_source_mismatch_stops_before_retrieval(self):
        tool = self._make_tool(MULTI_TARGET_CONFIG)

        with patch.object(tool, "_vector_search") as vector_search:
            result = tool.execute(
                {
                    "query": "Extract dosage from [Scanned_Prescription_Image_01]",
                    "target": "patient_context",
                    "source": "infective_endocarditis_extreme_embolism.pdf",
                    "source_slug": "infective_endocarditis_extreme_embolism",
                }
            )

        vector_search.assert_not_called()
        assert result["found"] is False
        assert result["empty"]["code"] == "source_mismatch"

    def test_matching_source_reference_is_removed_from_semantic_query(self):
        tool = self._make_tool(MULTI_TARGET_CONFIG)

        with (
            patch.object(tool, "_vector_search", return_value=[]) as vector_search,
            patch.object(tool, "_graph_context", return_value=[]),
        ):
            result = tool.execute(
                {
                    "query": "Extract dosage from [Adobe Scan 24 Jul 2026]",
                    "target": "patient_context",
                    "source": "Adobe Scan 24 Jul 2026.pdf",
                    "source_slug": "Adobe Scan 24 Jul 2026",
                }
            )

        assert vector_search.call_args.args[0] == "Extract dosage from"
        assert result["empty"]["code"] == "source_not_indexed"

    def test_patient_vector_search_filters_exact_source_and_scope(self):
        tool = self._make_tool(MULTI_TARGET_CONFIG)
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        hit = MagicMock()
        hit.score = 0.81
        hit.payload = {
            "source": "patient.pdf",
            "scope": "patient_context",
            "content": "Medication evidence.",
            "chunk_index": 4,
        }
        mock_client.query_points.return_value.points = [hit]
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.0] * 384
        tool._qdrant = mock_client
        tool._embedder = mock_embedder

        target = tool._target_config("patient_context")
        results = tool._vector_search(
            "dosage",
            fetch_limit=5,
            target=target,
            source="patient.pdf",
        )

        query_call = mock_client.query_points.call_args
        assert query_call.kwargs["collection_name"] == "patient_context"
        conditions = query_call.kwargs["query_filter"].must
        assert [(condition.key, condition.match.value) for condition in conditions] == [
            ("scope", "patient_context"),
            ("source", "patient.pdf"),
        ]
        assert results[0]["collection"] == "patient_context"

    def test_patient_graph_search_filters_scope_and_source_slug(self):
        tool = self._make_tool(MULTI_TARGET_CONFIG)
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        session.run.return_value = []
        tool._driver = driver

        tool._graph_context(
            ["dosage"],
            limit=5,
            target=tool._target_config("patient_context"),
            graph_source="patient_chunks",
        )

        call = session.run.call_args
        assert call.kwargs["scope"] == "patient_context"
        assert call.kwargs["source"] == "patient_chunks"


# ---------------------------------------------------------------------------
# TTL cache (BaseTool.cached_execute)
# ---------------------------------------------------------------------------

class TestCaching:
    def test_cache_prevents_double_execution(self):
        tool = GraphRAGTool(BASE_CONFIG)
        call_count = 0

        def fake_execute(query):
            nonlocal call_count
            call_count += 1
            return {"found": True, "vector_results": [], "graph_facts": []}

        tool.execute = fake_execute  # type: ignore[method-assign]
        tool.cached_execute("same query")
        tool.cached_execute("same query")
        assert call_count == 1

    def test_different_queries_execute_separately(self):
        tool = GraphRAGTool(BASE_CONFIG)
        call_count = 0

        def fake_execute(query):
            nonlocal call_count
            call_count += 1
            return {"found": False, "vector_results": [], "graph_facts": []}

        tool.execute = fake_execute  # type: ignore[method-assign]
        tool.cached_execute("query A")
        tool.cached_execute("query B")
        assert call_count == 2

    def test_mapping_key_order_does_not_split_cache_entries(self):
        tool = GraphRAGTool(BASE_CONFIG)
        call_count = 0

        def fake_execute(query):
            nonlocal call_count
            call_count += 1
            return {"found": False, "vector_results": [], "graph_facts": []}

        tool.execute = fake_execute  # type: ignore[method-assign]
        tool.cached_execute({"query": "same", "target": "literature"})
        tool.cached_execute({"target": "literature", "query": "same"})

        assert call_count == 1
