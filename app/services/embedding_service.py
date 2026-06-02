# ==========================================
# EMBEDDING SERVICE INTERFACE
# ==========================================
#
# Abstract interface for embedding models.
# Concrete implementations (ChromaDB, OpenAI,
# etc.) will be added in a future iteration.
#
# ==========================================


class EmbeddingService:

    """
    Abstract embedding service.

    Currently a no-op interface. Concrete
    implementations should override
    embed_text() and embed_batch().
    """

    async def embed_text(
        self,
        text: str
    ):
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            Embedding vector (list of floats).

        To be implemented in a subclass.
        """
        pass

    async def embed_batch(
        self,
        texts: list
    ):
        """
        Embed a batch of text strings.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        To be implemented in a subclass.
        """
        pass

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension.

        To be overridden in a subclass.
        """
        return 0
