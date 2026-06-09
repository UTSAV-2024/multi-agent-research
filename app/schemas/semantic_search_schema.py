# ==========================================
# SEMANTIC SEARCH SCHEMAS
# ==========================================
#
# Request/response models for the semantic
# retrieval API endpoint.
#
# ==========================================

from pydantic import BaseModel, Field


class SemanticSearchRequest(BaseModel):
    """Request payload for semantic search."""

    query: str = Field(
        ...,
        min_length=1,
        description="Search query text",
        examples=["German scientists after WWII"],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return",
        examples=[5],
    )


class SemanticSearchResult(BaseModel):
    """A single result from a semantic search."""

    chunk_id: int = Field(
        ...,
        description="Chunk identifier within the source",
    )

    content: str = Field(
        ...,
        description="Text content of the chunk",
    )

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (0 = no match, 1 = exact match)",
    )

    source_title: str = Field(
        ...,
        description="Title of the source document",
    )

    source_url: str = Field(
        ...,
        description="URL of the source document",
    )

    report_id: str = Field(
        ...,
        description="ID of the research report this chunk belongs to",
    )


class SemanticSearchResponse(BaseModel):
    """Response from a semantic search."""

    query: str = Field(
        ...,
        description="The original search query",
    )

    count: int = Field(
        ...,
        ge=0,
        description="Number of results returned",
    )

    results: list[SemanticSearchResult] = Field(
        default_factory=list,
        description="List of search results",
    )
