# ==========================================
# EMBEDDING SERVICE
# ==========================================
#
# Converts text chunks into vector embeddings
# using sentence-transformers.
#
# Model is loaded once in __init__ and reused
# for all subsequent calls. A module-level
# singleton (get_embedding_service) ensures
# the model is loaded only once per process.
#
# ==========================================

import time

from typing import List, Optional

from sentence_transformers import SentenceTransformer

from app.config.settings import settings
from app.utils.logger import logger


class EmbeddingService:
    """
    Embedding service powered by sentence-transformers.

    Usage:
        service = EmbeddingService()
        vec = service.embed_text("some text")
        vecs = service.embed_texts(["text1", "text2", ...])
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Load the embedding model **once** at initialisation.

        Args:
            model_name: HuggingFace model name or path.
                        Defaults to settings.EMBEDDING_MODEL.
            batch_size: Batch size for encode calls.
                        Defaults to settings.EMBEDDING_BATCH_SIZE.
        """
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._load_time = 0.0

        logger.info(
            f"[EMBEDDINGS] Loading model '{self._model_name}'..."
        )

        load_start = time.time()

        self._model = SentenceTransformer(self._model_name)

        self._dimension = self._model.get_embedding_dimension()
        self._load_time = round(time.time() - load_start, 2)

        logger.info(
            f"[EMBEDDINGS] Model loaded successfully"
        )

        logger.info(
            f"[EMBEDDINGS] Dimension={self._dimension}"
        )

    # ---------------------------------------------------------------
    # PROPERTIES
    # ---------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The embedding model identifier."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        return self._dimension

    @property
    def load_time(self) -> float:
        """Time taken to load the model (seconds)."""
        return self._load_time

    # ---------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: The input text.

        Returns:
            A list of floats representing the embedding vector.
        """
        logger.info(
            f"[EMBEDDINGS] Embedding single text "
            f"(len={len(text)} chars)"
        )

        vector = self._model.encode(text).tolist()

        logger.info(
            f"[EMBEDDINGS] Generated embedding "
            f"(dim={len(vector)})"
        )

        return vector

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts.

        Uses batched encoding for performance.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors (list of list of floats).
        """
        if not texts:
            logger.warning(
                "[EMBEDDINGS] embed_texts called with empty list"
            )
            return []

        logger.info(
            f"[EMBEDDINGS] Generating embeddings for "
            f"{len(texts)} texts "
            f"(batch_size={self._batch_size})"
        )

        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
        ).tolist()

        logger.info(
            f"[EMBEDDINGS] Generated {len(vectors)} embeddings"
        )

        return vectors


# ==========================================
# MODULE-LEVEL SINGLETON
# ==========================================
#
# Loaded once per process so the model
# is reused across all workflow invocations.
# Use get_embedding_service() anywhere in the
# codebase to access the shared instance.
#
# ==========================================

_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Return the shared EmbeddingService singleton, loading it
    on first call."""
    global _embedding_service
    if _embedding_service is not None:
        logger.info(
            "[EMBEDDINGS] Reusing loaded model"
        )
        return _embedding_service
    _embedding_service = EmbeddingService()
    return _embedding_service
