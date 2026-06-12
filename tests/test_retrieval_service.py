# ==========================================
# RETRIEVAL SERVICE TESTS
# ==========================================
#
# Tests for the retrieval_service module.
#
# ==========================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock


class TestRetrieveChunks:
    """Tests for retrieve_chunks()."""

    @patch("app.services.retrieval_service.VectorStore")
    def test_returns_query_results(self, mock_store_class):
        """retrieve_chunks returns ChromaDB-style results."""
        from app.services.retrieval_service import retrieve_chunks

        mock_instance = MagicMock()
        mock_store_class.return_value = mock_instance

        # Mock semantic query results
        mock_instance.query.return_value = {
            "ids": [["rep1:src1:1", "rep1:src1:2"]],
            "documents": [["Doc one.", "Doc two."]],
            "metadatas": [[
                {"chunk_id": 1, "url": "https://a.com", "title": "A"},
                {"chunk_id": 2, "url": "https://b.com", "title": "B"},
            ]],
            "distances": [[0.15, 0.42]],
        }

        # Mock keyword retrieval (all documents)
        mock_instance._collection.get.return_value = {
            "ids": ["rep1:src1:1", "rep1:src1:2"],
            "documents": ["Doc one.", "Doc two."],
            "metadatas": [
                {"chunk_id": 1, "url": "https://a.com", "title": "A"},
                {"chunk_id": 2, "url": "https://b.com", "title": "B"},
            ],
        }

        result = retrieve_chunks(query="test query", top_k=2)

        # Should return ChromaDB-style dict
        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        assert "distances" in result

    @patch("app.services.retrieval_service.VectorStore")
    def test_calls_query_with_correct_params(self, mock_store_class):
        """VectorStore.query() is called with the right arguments."""
        from app.services.retrieval_service import retrieve_chunks

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

        retrieve_chunks(query="machine learning", top_k=3)

        mock_instance.query.assert_called_once_with(
            query_text="machine learning",
            top_k=9,  # top_k * HYBRID_RETRIEVAL_MULTIPLIER (3)
        )

    @patch("app.services.retrieval_service.VectorStore")
    def test_initializes_collection(self, mock_store_class):
        """VectorStore.initialize_collection() is called."""
        from app.services.retrieval_service import retrieve_chunks

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

        retrieve_chunks(query="test", top_k=1)

        mock_instance.initialize_collection.assert_called_once()

    @patch("app.services.retrieval_service.VectorStore")
    def test_closes_store(self, mock_store_class):
        """VectorStore.close() is called after query."""
        from app.services.retrieval_service import retrieve_chunks

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

        retrieve_chunks(query="test", top_k=1)

        mock_instance.close.assert_called_once()

    @patch("app.services.retrieval_service.VectorStore")
    def test_closes_store_on_error(self, mock_store_class):
        """VectorStore.close() is called even when query raises."""
        from app.services.retrieval_service import retrieve_chunks

        mock_instance = MagicMock()
        mock_store_class.return_value = mock_instance
        mock_instance.query.side_effect = RuntimeError("DB error")

        import pytest
        with pytest.raises(RuntimeError):
            retrieve_chunks(query="test", top_k=1)

        mock_instance.close.assert_called_once()

    @patch("app.services.retrieval_service.VectorStore")
    def test_deterministic_ordering(self, mock_store_class):
        """Results are sorted deterministically (by score desc, then stable_id)."""
        from app.services.retrieval_service import retrieve_chunks

        mock_instance = MagicMock()
        mock_store_class.return_value = mock_instance

        mock_instance.query.return_value = {
            "ids": [["r:s:2", "r:s:1"]],
            "documents": [["Chunk B.", "Chunk A."]],
            "metadatas": [[
                {"chunk_id": 2, "url": "https://b.com"},
                {"chunk_id": 1, "url": "https://a.com"},
            ]],
            "distances": [[0.5, 0.5]],
        }
        mock_instance._collection.get.return_value = {
            "ids": ["r:s:1", "r:s:2"],
            "documents": ["Chunk A.", "Chunk B."],
            "metadatas": [
                {"chunk_id": 1, "url": "https://a.com"},
                {"chunk_id": 2, "url": "https://b.com"},
            ],
        }

        result = retrieve_chunks(query="test", top_k=2)

        ids = result["ids"][0]
        # With equal scores, should sort by stable_id ascending
        assert ids[0] < ids[1], (
            f"Expected deterministic ordering, got: {ids}"
        )
