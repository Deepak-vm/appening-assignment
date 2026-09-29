"""Tests for chunker output, threshold routing, and API schema."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

class TestChunker:
    """Verify chunk_pages produces correct structure and deterministic IDs."""

    def test_chunks_have_page_metadata(self):
        from rag.ingest import chunk_pages

        pages = [(1, "A" * 1000), (2, "B" * 1000)]
        chunks = chunk_pages(pages, source_name="test.pdf")
        assert all("page" in c for c in chunks)
        assert all(c["page"] in {1, 2} for c in chunks)

    def test_chunk_ids_are_deterministic(self):
        from rag.ingest import chunk_pages

        pages = [(1, "Hello world. " * 100)]
        ids_first = [c["id"] for c in chunk_pages(pages, source_name="test.pdf")]
        ids_second = [c["id"] for c in chunk_pages(pages, source_name="test.pdf")]
        assert ids_first == ids_second

    def test_chunk_ids_are_unique(self):
        from rag.ingest import chunk_pages

        pages = [(1, "A" * 2000)]
        chunks = chunk_pages(pages, source_name="test.pdf")
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_text_yields_no_chunks(self):
        from rag.ingest import chunk_pages

        chunks = chunk_pages([(1, "   ")], source_name="test.pdf")
        assert chunks == []


# ---------------------------------------------------------------------------
# Threshold routing tests
# ---------------------------------------------------------------------------

class TestRelevanceRouting:
    """Verify route_by_relevance routes correctly around the threshold."""

    def _state(self, scores: list[float]) -> dict:
        chunks = [{"text": "t", "page": 1, "score": s} for s in scores]
        return {"question": "test", "chunks": chunks, "answer": "", "confidence": 0.0}

    def test_routes_to_generate_above_threshold(self):
        from rag.graph import route_by_relevance

        assert route_by_relevance(self._state([0.9, 0.8, 0.7])) == "generate"

    def test_routes_to_fallback_below_threshold(self):
        from rag.graph import route_by_relevance

        assert route_by_relevance(self._state([0.1, 0.05, 0.02])) == "fallback"

    def test_routes_to_fallback_on_empty_chunks(self):
        from rag.graph import route_by_relevance

        assert route_by_relevance(self._state([])) == "fallback"

    def test_boundary_exactly_at_threshold(self):
        """Score equal to threshold should route to generate."""
        from rag.config import settings
        from rag.graph import route_by_relevance

        assert route_by_relevance(self._state([settings.relevance_threshold])) == "generate"


# ---------------------------------------------------------------------------
# API schema tests (graph mocked)
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    """Integration tests for POST /chat with rag_graph mocked."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        mock_state = {
            "answer": "Agentic AI systems use autonomous agents (p. 5).",
            "confidence": 0.82,
            "chunks": [{"text": "Some context text.", "page": 5, "score": 0.82}],
        }

        with patch("rag.api.rag_graph") as mock_graph:
            mock_graph.invoke.return_value = mock_state
            from rag.api import app
            yield TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_chat_returns_correct_schema(self, client):
        r = client.post("/chat", json={"question": "What is agentic AI?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_chat_source_has_required_fields(self, client):
        r = client.post("/chat", json={"question": "What is agentic AI?"})
        source = r.json()["sources"][0]
        assert "text" in source
        assert "page" in source
        assert "score" in source

    def test_empty_question_is_rejected(self, client):
        r = client.post("/chat", json={"question": ""})
        assert r.status_code == 422
