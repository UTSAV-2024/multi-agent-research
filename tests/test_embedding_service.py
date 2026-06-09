# ==========================================
# EMBEDDING SERVICE TESTS
# ==========================================
#
# Tests the EmbeddingService and its
# integration with VectorStore.
#
# ==========================================

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.embedding_service import (
    EmbeddingService,
    get_embedding_service,
)
from app.services.vector_store import (
    VectorStore,
    make_stable_id,
)


class TestEmbeddingService:
    """Tests for EmbeddingService in isolation."""

    def setup_method(self):
        self.service = EmbeddingService()

    # ------------------------------------------------------------------
    # TESTS
    # ------------------------------------------------------------------

    def test_embed_text_returns_list_of_floats(self):
        """embed_text() returns a list of floats."""
        vector = self.service.embed_text(
            "Operation Paperclip recruited German scientists after WWII."
        )
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(v, float) for v in vector)

    def test_embed_text_has_384_dimensions(self):
        """all-MiniLM-L6-v2 produces 384-dimensional vectors."""
        vector = self.service.embed_text(
            "Artificial intelligence is transforming the world."
        )
        assert len(vector) == 384

    def test_embed_texts_returns_same_count(self):
        """embed_texts() returns same number of vectors as inputs."""
        texts = [
            "Cats are feline animals.",
            "Dogs are canine animals.",
            "Quantum mechanics studies subatomic particles.",
            "Machine learning is a subset of AI.",
        ]
        vectors = self.service.embed_texts(texts)
        assert len(vectors) == 4
        assert all(len(v) == 384 for v in vectors)

    def test_embed_texts_empty_input(self):
        """embed_texts([]) returns empty list."""
        vectors = self.service.embed_texts([])
        assert vectors == []

    def test_embed_texts_maintains_order(self):
        """The order of input texts is preserved in output vectors."""
        texts = [
            "First document about Python programming.",
            "Second document about Rust systems programming.",
        ]
        vectors = self.service.embed_texts(texts)

        # Re-embed individually to compare
        first = self.service.embed_text(texts[0])
        second = self.service.embed_text(texts[1])

        # Use approximate comparison (batched vs single encoding
        # can produce tiny floating-point differences)
        for v1, v2 in zip(vectors[0], first):
            assert abs(v1 - v2) < 1e-5, (
                f"Batched and single embeddings differ at position"
            )
        for v1, v2 in zip(vectors[1], second):
            assert abs(v1 - v2) < 1e-5

    def test_model_name_property(self):
        """model_name returns the configured model."""
        assert "all-MiniLM-L6-v2" in self.service.model_name

    def test_dimension_property(self):
        """dimension property returns 384."""
        assert self.service.dimension == 384

    # ------------------------------------------------------------------
    # SINGLETON / LIFECYCLE TESTS
    # ------------------------------------------------------------------

    def test_get_embedding_service_returns_singleton(self):
        """Multiple calls to get_embedding_service() return the same instance."""
        svc1 = get_embedding_service()
        svc2 = get_embedding_service()
        assert svc1 is svc2, (
            "get_embedding_service() must return the same instance "
            "on subsequent calls"
        )

    def test_get_embedding_service_load_time_recorded(self):
        """After first load, load_time is a positive number."""
        svc = get_embedding_service()
        assert svc.load_time > 0, (
            f"Expected load_time > 0, got {svc.load_time}"
        )

    def test_get_embedding_service_dimension_correct(self):
        """Singleton service reports correct dimension."""
        svc = get_embedding_service()
        assert svc.dimension == 384


class TestEmbeddingVectorStoreIntegration:
    """Integration tests: EmbeddingService + VectorStore end-to-end."""

    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(
            persist_dir=self._tmpdir,
            collection_name="test_embed_integration",
        )
        self.vector_store.initialize_collection()

    def teardown_method(self):
        self.vector_store.close()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_embed_and_store_10_chunks(self):
        """Insert 10 chunks with embeddings, verify 10 vectors exist."""
        report_id = "embed_test_report"
        source_id = "src_integration"
        chunks = [
            {
                "chunk_id": i + 1,
                "content": f"This is chunk {i + 1} discussing "
                           f"topic number {i + 1} in machine learning.",
                "title": f"Article {i + 1}",
                "url": f"https://example.com/{i + 1}",
                "start": 0,
                "end": 80,
            }
            for i in range(10)
        ]

        # Generate embeddings
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        assert len(embeddings) == 10
        assert all(len(e) == 384 for e in embeddings)

        # Store in ChromaDB
        added = self.vector_store.add_documents(
            report_id=report_id,
            source_id=source_id,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.dimension,
        )

        assert added == 10

        # Verify vectors exist by querying
        results = self.vector_store.query(
            "machine learning topics",
            top_k=10,
        )

        assert len(results["ids"][0]) == 10

        # Verify embedding metadata was stored
        for metadata in results["metadatas"][0]:
            assert metadata["embedding_model"] == self.embedding_service.model_name
            assert metadata["embedding_dimension"] == 384

    def test_embed_and_store_without_embeddings_fallback(self):
        """Fallback: add_documents works without pre-computed embeddings."""
        report_id = "fallback_test"
        source_id = "src_fallback"
        chunks = [
            {
                "chunk_id": 1,
                "content": "ChromaDB computes embeddings automatically "
                           "when none are provided.",
                "title": "Fallback Test",
                "url": "https://example.com/fallback",
                "start": 0,
                "end": 80,
            }
        ]

        # This should work even without passing embeddings
        # (ChromaDB computes them internally)
        added = self.vector_store.add_documents(
            report_id=report_id,
            source_id=source_id,
            chunks=chunks,
        )

        assert added == 1

        results = self.vector_store.query(
            "automatic embeddings",
            top_k=5,
        )

        assert len(results["ids"][0]) == 1
