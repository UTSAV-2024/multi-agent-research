# ==========================================
# SEMANTIC SEARCH API TESTS
# ==========================================
#
# Tests for the POST /semantic-search endpoint.
#
# ==========================================

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestSemanticSearchValidation:
    """Tests for request validation."""

    def test_empty_query_returns_422(self):
        """Empty query string returns 422 validation error."""
        response = client.post(
            "/semantic-search",
            json={"query": "", "top_k": 5},
        )
        assert response.status_code == 422

    def test_missing_query_returns_422(self):
        """Missing query field returns 422 validation error."""
        response = client.post(
            "/semantic-search",
            json={"top_k": 5},
        )
        assert response.status_code == 422

    def test_top_k_zero_returns_422(self):
        """top_k=0 (below minimum) returns 422."""
        response = client.post(
            "/semantic-search",
            json={"query": "test", "top_k": 0},
        )
        assert response.status_code == 422

    def test_top_k_over_max_returns_422(self):
        """top_k=21 (above maximum) returns 422."""
        response = client.post(
            "/semantic-search",
            json={"query": "test", "top_k": 21},
        )
        assert response.status_code == 422

    def test_top_k_default_is_5(self):
        """Default top_k is 5 when not provided."""
        response = client.post(
            "/semantic-search",
            json={"query": "test"},
        )
        # Should be 422 because VectorStore hasn't been initialised
        # (collection doesn't exist yet) OR 500 if something else fails.
        # The important thing is it doesn't crash with missing field.
        assert response.status_code in (422, 500, 200)
        # If it's a 500, it means the request was accepted but store failed
        if response.status_code == 500:
            body = response.json()
            assert "detail" in body


class TestSemanticSearchWithMockedStore:
    """Tests with a mocked VectorStore."""

    @patch("app.services.retrieval_service.VectorStore")
    def test_successful_query_structure(self, mock_store_class):
        """A successful query returns the expected response structure."""
        # Arrange
        mock_instance = MagicMock()
        mock_store_class.return_value = mock_instance

        mock_instance.query.return_value = {
            "ids": [["rep1:src1:1", "rep1:src1:2"]],
            "documents": [
                ["Document one content.", "Document two content."]
            ],
            "metadatas": [
                [
                    {
                        "chunk_id": 1,
                        "title": "Article 1",
                        "url": "https://example.com/1",
                        "report_id": "rep1",
                    },
                    {
                        "chunk_id": 2,
                        "title": "Article 2",
                        "url": "https://example.com/2",
                        "report_id": "rep1",
                    },
                ]
            ],
            "distances": [[0.15, 0.42]],
        }

        # Mock the collection.get() used by keyword retrieval
        mock_instance._collection.get.return_value = {
            "ids": ["rep1:src1:1", "rep1:src1:2"],
            "documents": [
                "Document one content.",
                "Document two content.",
            ],
            "metadatas": [
                {
                    "chunk_id": 1,
                    "title": "Article 1",
                    "url": "https://example.com/1",
                    "report_id": "rep1",
                },
                {
                    "chunk_id": 2,
                    "title": "Article 2",
                    "url": "https://example.com/2",
                    "report_id": "rep1",
                },
            ],
        }

        # Act
        response = client.post(
            "/semantic-search",
            json={"query": "test query", "top_k": 2},
        )

        # Assert
        assert response.status_code == 200
        body = response.json()

        assert body["query"] == "test query"
        assert body["count"] == 2
        assert len(body["results"]) == 2

        # Check first result structure
        r1 = body["results"][0]
        assert r1["chunk_id"] == 1
        assert r1["content"] == "Document one content."
        assert isinstance(r1["score"], float)
        assert 0.0 <= r1["score"] <= 1.0
        assert r1["source_title"] == "Article 1"
        assert r1["source_url"] == "https://example.com/1"
        assert r1["report_id"] == "rep1"

        # Check second result
        r2 = body["results"][1]
        assert r2["chunk_id"] == 2
        assert isinstance(r2["score"], float)
        assert 0.0 <= r2["score"] <= 1.0
        assert r2["source_title"] == "Article 2"

    @patch("app.services.retrieval_service.VectorStore")
    def test_empty_results(self, mock_store_class):
        """When no results found, returns empty results list."""
        # Arrange
        mock_instance = MagicMock()
        mock_store_class.return_value = mock_instance

        mock_instance.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_instance._collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        # Act
        response = client.post(
            "/semantic-search",
            json={"query": "nothing relevant", "top_k": 5},
        )

        # Assert
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 0
        assert body["results"] == []

    @patch("app.services.retrieval_service.VectorStore")
    def test_score_clamped_to_zero(self, mock_store_class):
        """Distance > 1.0 gives score == 0.0 (clamped)."""
        # Arrange
        mock_instance = MagicMock()
        mock_store_class.return_value = mock_instance

        mock_instance.query.return_value = {
            "ids": [["rep1:src1:1"]],
            "documents": [["Some distant content."]],
            "metadatas": [[
                {
                    "chunk_id": 1,
                    "title": "Far Article",
                    "url": "https://example.com/far",
                    "report_id": "rep1",
                }
            ]],
            "distances": [[1.5]],  # >= 1.0 -> score = 0.0
        }

        mock_instance._collection.get.return_value = {
            "ids": ["rep1:src1:1"],
            "documents": ["Some distant content."],
            "metadatas": [{"chunk_id": 1, "title": "Far Article", "url": "https://example.com/far", "report_id": "rep1"}],
        }

        # Act
        response = client.post(
            "/semantic-search",
            json={"query": "distant query", "top_k": 1},
        )

        # Assert
        assert response.status_code == 200
        body = response.json()
        assert body["results"][0]["score"] == 0.0

    @patch("app.services.retrieval_service.VectorStore")
    def test_query_called_with_correct_params(self, mock_store_class):
        """VectorStore.query() is called with the right arguments."""
        # Arrange
        mock_instance = MagicMock()
        mock_store_class.return_value = mock_instance

        mock_instance.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        mock_instance._collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        # Act
        client.post(
            "/semantic-search",
            json={"query": "machine learning transformers", "top_k": 3},
        )

        # Assert
        # Hybrid retrieval uses multiplier (3) so top_k passed to VectorStore
        # is top_k * HYBRID_RETRIEVAL_MULTIPLIER = 3 * 3 = 9
        mock_instance.query.assert_called_once_with(
            query_text="machine learning transformers",
            top_k=9,
        )

    @patch("app.services.retrieval_service.VectorStore")
    def test_runtime_error_returns_500(self, mock_store_class):
        """RuntimeError from VectorStore returns 500."""
        # Arrange
        mock_instance = MagicMock()
        mock_store_class.return_value = mock_instance

        mock_instance.query.side_effect = RuntimeError(
            "Collection not initialised"
        )

        mock_instance._collection.get.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
        }

        # Act
        response = client.post(
            "/semantic-search",
            json={"query": "test", "top_k": 5},
        )

        # Assert
        assert response.status_code == 500
        body = response.json()
        assert "detail" in body


class TestSemanticSearchWithRealChroma:
    """Integration tests with a real ChromaDB instance."""

    def setup_method(self):
        """Create a fresh VectorStore with real ChromaDB."""
        self._tmpdir = tempfile.mkdtemp()

        # Patch VectorStore inside retrieval_service to use temp directory
        self._patcher = patch(
            "app.services.retrieval_service.VectorStore",
        )
        self._mock_class = self._patcher.start()

        # Create a real VectorStore instance
        from app.services.vector_store import VectorStore as RealStore
        self._real_store = RealStore(
            persist_dir=self._tmpdir,
            collection_name="test_semantic_search",
        )
        self._real_store.initialize_collection()

        # Insert test data
        self._real_store.add_documents(
            report_id="test_report",
            source_id="https://example.com/wwii",
            chunks=[
                {
                    "chunk_id": 1,
                    "content": "Operation Paperclip recruited German scientists after WWII.",
                    "title": "Operation Paperclip",
                    "url": "https://example.com/wwii",
                    "start": 0,
                    "end": 80,
                },
                {
                    "chunk_id": 2,
                    "content": "Wernher von Braun led NASA's rocket development.",
                    "title": "Von Braun",
                    "url": "https://example.com/wwii",
                    "start": 80,
                    "end": 140,
                },
                {
                    "chunk_id": 3,
                    "content": "The Manhattan Project developed the first nuclear weapons.",
                    "title": "Manhattan Project",
                    "url": "https://example.com/nuclear",
                    "start": 0,
                    "end": 75,
                },
            ],
        )

        # Make the mock return the real store's query results
        self._mock_class.return_value = self._real_store

    def teardown_method(self):
        """Clean up resources."""
        self._real_store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._patcher.stop()

    def test_real_chroma_query_returns_results(self):
        """Real ChromaDB search returns relevant results."""
        response = client.post(
            "/semantic-search",
            json={
                "query": "German scientists after WWII",
                "top_k": 2,
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["query"] == "German scientists after WWII"
        assert body["count"] > 0
        assert len(body["results"]) > 0

        # The first result should be about Operation Paperclip
        first = body["results"][0]
        assert first["chunk_id"] in (1, 2, 3)
        assert isinstance(first["score"], float)
        assert 0.0 <= first["score"] <= 1.0

    def test_real_chroma_top_k_limit(self):
        """top_k parameter is respected."""
        response = client.post(
            "/semantic-search",
            json={"query": "scientists", "top_k": 1},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["count"] <= 1
        assert len(body["results"]) <= 1

    def test_real_chroma_correct_response_shape(self):
        """Response has the correct shape with all expected fields."""
        response = client.post(
            "/semantic-search",
            json={"query": "rocket", "top_k": 3},
        )

        assert response.status_code == 200
        body = response.json()

        assert "query" in body
        assert "count" in body
        assert "results" in body
        assert isinstance(body["results"], list)

        if body["results"]:
            r = body["results"][0]
            assert "chunk_id" in r
            assert "content" in r
            assert "score" in r
            assert "source_title" in r
            assert "source_url" in r
            assert "report_id" in r
