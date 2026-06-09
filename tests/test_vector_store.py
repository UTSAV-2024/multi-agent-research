# ==========================================
# VECTOR STORE INTEGRATION TEST
# ==========================================
#
# Tests the full ChromaDB-backed VectorStore
# lifecycle: insert, query, delete.
#
# ==========================================

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.vector_store import (
    VectorStore,
    make_stable_id,
)


class TestVectorStore:
    """Integration tests for VectorStore with real ChromaDB."""

    def setup_method(self):
        """Create a fresh VectorStore in a temp directory per test."""
        self._tmpdir = tempfile.mkdtemp()
        self.store = VectorStore(
            persist_dir=self._tmpdir,
            collection_name="test_research_chunks",
        )
        self.store.initialize_collection()

    def teardown_method(self):
        """Close ChromaDB client, then clean up the temp directory."""
        self.store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunks(count: int = 10) -> list:
        """Generate test chunks with varied content."""
        topics = [
            "artificial intelligence and machine learning",
            "deep neural networks for image recognition",
            "natural language processing with transformers",
            "reinforcement learning for robotics",
            "computer vision in autonomous vehicles",
            "large language models and their applications",
            "recommender systems for e-commerce",
            "time series forecasting with LSTMs",
            "generative adversarial networks for image synthesis",
            "attention mechanisms in sequence models",
        ]
        return [
            {
                "chunk_id": i + 1,
                "content": f"Chunk {i + 1}: This document discusses {topics[i]}.",
                "title": f"Source Article {i + 1}",
                "url": f"https://example.com/article-{i + 1}",
                "start": 0,
                "end": 80,
            }
            for i in range(count)
        ]

    # ------------------------------------------------------------------
    # TESTS
    # ------------------------------------------------------------------

    def test_make_stable_id_format(self):
        """Stable ID follows the report_id:source_id:chunk_id pattern."""
        stable_id = make_stable_id("rep123", "src456", 7)
        assert stable_id == "rep123:src456:7"

    def test_insert_and_query(self):
        """Insert 10 chunks, query with top_k=3, verify 3 results."""
        report_id = "test_report_001"
        source_id = "src_article_1"
        chunks = self._make_chunks(10)

        added = self.store.add_documents(report_id, source_id, chunks)
        assert added == 10

        results = self.store.query(
            "artificial intelligence machine learning",
            top_k=3,
        )

        # ChromaDB returns lists-of-lists (one per query text)
        assert len(results["ids"][0]) == 3
        assert len(results["documents"][0]) == 3
        assert len(results["metadatas"][0]) == 3
        assert len(results["distances"][0]) == 3

        # Verify stable ID format in results
        for doc_id in results["ids"][0]:
            parts = doc_id.split(":")
            assert len(parts) == 3
            assert parts[0] == report_id
            assert parts[1] == source_id

    def test_delete_by_report_removes_all(self):
        """Insert chunks, delete by report, verify 0 results remain."""
        report_id = "test_report_002"
        source_id = "src_article_2"
        chunks = self._make_chunks(10)

        self.store.add_documents(report_id, source_id, chunks)

        # Verify docs are there before deletion
        before = self.store.query("machine learning", top_k=10)
        assert len(before["ids"][0]) > 0

        # Delete
        deleted_count = self.store.delete_by_report(report_id)
        assert deleted_count == 10

        # Verify 0 results remain
        after = self.store.query("machine learning", top_k=10)
        assert len(after["ids"][0]) == 0
        assert len(after["documents"][0]) == 0
        assert len(after["metadatas"][0]) == 0

    def test_insert_multiple_sources_same_report(self):
        """Multiple sources within the same report are all queryable."""
        report_id = "test_report_003"
        source_a = "https://example.com/a"
        source_b = "https://example.com/b"

        chunks_a = [
            {"chunk_id": 1, "content": "Python is a programming language.",
             "title": "Python", "url": source_a, "start": 0, "end": 40},
            {"chunk_id": 2, "content": "Python supports multiple paradigms.",
             "title": "Python", "url": source_a, "start": 40, "end": 80},
        ]
        chunks_b = [
            {"chunk_id": 1, "content": "Rust is a systems language.",
             "title": "Rust", "url": source_b, "start": 0, "end": 40},
            {"chunk_id": 2, "content": "Rust guarantees memory safety.",
             "title": "Rust", "url": source_b, "start": 40, "end": 80},
        ]

        self.store.add_documents(report_id, source_a, chunks_a)
        self.store.add_documents(report_id, source_b, chunks_b)

        # Query for Python content
        py_results = self.store.query("programming language", top_k=5)
        py_ids = py_results["ids"][0]
        py_sources = [m["source_id"] for m in py_results["metadatas"][0]]
        assert source_a in py_sources

        # Delete entire report
        deleted = self.store.delete_by_report(report_id)
        assert deleted == 4

        after = self.store.query("programming", top_k=5)
        assert len(after["ids"][0]) == 0

    def test_query_returns_relevant_results(self):
        """Query returns the most semantically relevant chunks."""
        report_id = "test_report_004"
        source_id = "https://example.com/relevance"

        chunks = [
            {"chunk_id": 1, "content": "Cats are feline animals.",
             "title": "Cats", "url": source_id, "start": 0, "end": 30},
            {"chunk_id": 2, "content": "Dogs are canine animals.",
             "title": "Dogs", "url": source_id, "start": 0, "end": 30},
            {"chunk_id": 3, "content": "Quantum mechanics studies subatomic particles.",
             "title": "Quantum", "url": source_id, "start": 0, "end": 55},
            {"chunk_id": 4, "content": "Machine learning is a subset of AI.",
             "title": "ML", "url": source_id, "start": 0, "end": 40},
        ]

        self.store.add_documents(report_id, source_id, chunks)

        results = self.store.query("feline pets", top_k=2)
        docs = results["documents"][0]

        # The cat document should be among the top results
        cat_found = any("Cats are feline" in d for d in docs)
        assert cat_found, (
            f"Expected 'Cats are feline' in top results, got: {docs}"
        )
