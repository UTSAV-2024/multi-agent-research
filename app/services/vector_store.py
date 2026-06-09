# ==========================================
# VECTOR STORE SERVICE
# ==========================================
#
# ChromaDB-backed vector store for semantic
# retrieval of research chunks.
#
# Uses stable IDs (report_id:source_id:chunk_id)
# to prevent duplicate vectors.
#
# ==========================================

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config.settings import settings
from app.utils.logger import logger


def make_stable_id(
    report_id: str,
    source_id: str,
    chunk_id: int,
) -> str:
    """
    Build a stable, deterministic ID for a vector.

    Format: report_id:source_id:chunk_id

    This prevents duplicate vectors when the same
    chunk is re-processed.
    """
    return f"{report_id}:{source_id}:{chunk_id}"


class VectorStore:
    """
    ChromaDB-backed vector store for semantic retrieval.

    Usage:
        store = VectorStore()
        store.initialize_collection()
        store.add_documents(report_id, source_id, chunks)
        results = store.query("some query text", top_k=5)
        store.delete_by_report(report_id)
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            persist_dir: Directory for ChromaDB persistent storage.
                         Defaults to settings.CHROMA_PERSIST_DIR.
            collection_name: Name of the ChromaDB collection.
                             Defaults to settings.VECTOR_COLLECTION.
        """
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._collection_name = collection_name or settings.VECTOR_COLLECTION
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    # ---------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------

    def initialize_collection(self) -> None:
        """
        Create or retrieve the ChromaDB collection.

        Must be called before add_documents / query / delete.
        """
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
        )

        logger.info(
            f"[VECTOR STORE] Initialised collection "
            f"'{self._collection_name}' "
            f"at '{self._persist_dir}'"
        )

    def add_documents(
        self,
        report_id: str,
        source_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
        embedding_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
    ) -> int:
        """
        Add a list of chunks to the vector store.

        Each chunk is expected to have at least:
            chunk_id (int), content (str)

        When embeddings are provided they are passed directly to
        ChromaDB, bypassing its internal embedding function.

        Args:
            report_id: The parent report ID.
            source_id: The source/URL identifier.
            chunks: List of chunk dicts from chunking_service.
            embeddings: Optional pre-computed embeddings. Must be
                        the same length as chunks.
            embedding_model: Model name used to generate embeddings.
            embedding_dimension: Dimension of the embedding vectors.

        Returns:
            Number of documents added.
        """
        self._require_collection()

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            ids.append(make_stable_id(report_id, source_id, chunk_id))
            documents.append(chunk["content"])

            metadata: Dict[str, Any] = {
                "report_id": report_id,
                "source_id": source_id,
                "chunk_id": chunk_id,
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "start": chunk.get("start", 0),
                "end": chunk.get("end", 0),
            }

            # Embedding provenance metadata (useful for debugging)
            if embedding_model is not None:
                metadata["embedding_model"] = embedding_model
            if embedding_dimension is not None:
                metadata["embedding_dimension"] = embedding_dimension

            metadatas.append(metadata)

        kwargs: Dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }

        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        self._collection.add(**kwargs)

        logger.info(
            f"[VECTOR STORE] Added {len(ids)} documents "
            f"for report_id={report_id}, source_id={source_id}"
        )

        return len(ids)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Query the vector store for semantically similar chunks.

        Args:
            query_text: The search query.
            top_k: Number of results to return.

        Returns:
            ChromaDB query result dict with keys:
                ids, distances, metadatas, documents
        """
        self._require_collection()

        results = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
        )

        logger.info(
            f"[VECTOR STORE] Queried '{query_text[:50]}...' "
            f"top_k={top_k}, got "
            f"{len(results.get('ids', [[]])[0])} results"
        )

        return results

    def delete_by_report(self, report_id: str) -> int:
        """
        Delete all vectors belonging to a specific report.

        Args:
            report_id: The report ID whose vectors to delete.

        Returns:
            Number of deleted vectors (approximate; ChromaDB
            may report the count differently).
        """
        self._require_collection()

        # Count first for logging
        count_result = self._collection.get(
            where={"report_id": report_id},
        )
        count = len(count_result.get("ids", []))

        self._collection.delete(
            where={"report_id": report_id},
        )

        logger.info(
            f"[VECTOR STORE] Deleted {count} documents "
            f"for report_id={report_id}"
        )

        return count

    def close(self) -> None:
        """
        Release the ChromaDB client and free resources.

        Call this when the store is no longer needed (e.g. in
        test teardown) so the underlying SQLite connection is
        closed and the persistence directory can be cleaned up.
        """
        self._collection = None
        self._client = None

    # ---------------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------------

    def _require_collection(self) -> None:
        """Guard: raise if collection has not been initialised."""
        if self._collection is None:
            raise RuntimeError(
                "VectorStore not initialised. "
                "Call initialize_collection() first."
            )
