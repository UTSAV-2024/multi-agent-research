# ==========================================
# SEMANTIC SEARCH API
# ==========================================
#
# Public semantic retrieval endpoint backed
# by ChromaDB similarity search.
#
# ==========================================

from fastapi import APIRouter, HTTPException

from app.schemas.semantic_search_schema import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from app.services.vector_store import VectorStore
from app.utils.logger import logger


router = APIRouter()


@router.post(
    "/semantic-search",
    response_model=SemanticSearchResponse,
    summary="Semantic Search",
    description="Search stored research chunks by semantic similarity.",
    responses={
        200: {
            "description": "Successful search",
            "content": {
                "application/json": {
                    "example": {
                        "query": "German scientists after WWII",
                        "count": 2,
                        "results": [
                            {
                                "chunk_id": 1,
                                "content": "Operation Paperclip...",
                                "score": 0.85,
                                "source_title": "Article Title",
                                "source_url": "https://example.com/article",
                                "report_id": "report-uuid",
                            }
                        ],
                    }
                }
            },
        },
        422: {"description": "Validation error (empty query, invalid top_k)"},
        500: {"description": "Vector store not initialised or internal error"},
    },
)
async def semantic_search(
    payload: SemanticSearchRequest,
) -> SemanticSearchResponse:
    """
    Perform a semantic search over stored research chunks.

    The query is embedded and compared against all stored chunk
    vectors using ChromaDB similarity search. Results are returned
    with a normalised similarity score (0.0 – 1.0).
    """
    logger.info("[SEMANTIC SEARCH] Query received")

    # ------------------------------------------------------------------
    # Initialise vector store
    # ------------------------------------------------------------------

    try:
        vector_store = VectorStore()
        vector_store.initialize_collection()
    except Exception as e:
        logger.error(
            f"[SEMANTIC SEARCH] Failed to initialise vector store: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Vector store initialisation failed: {e}",
        )

    # ------------------------------------------------------------------
    # Execute query
    # ------------------------------------------------------------------

    try:
        results = vector_store.query(
            query_text=payload.query,
            top_k=payload.top_k,
        )
    except RuntimeError as e:
        logger.error(
            f"[SEMANTIC SEARCH] Vector store error: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"[SEMANTIC SEARCH] Query failed: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {e}",
        )
    finally:
        vector_store.close()

    # ------------------------------------------------------------------
    # Normalise results
    # ------------------------------------------------------------------

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    items: list[SemanticSearchResult] = []

    for i in range(len(ids)):
        # ChromaDB distance: 0 = identical, larger = less similar
        distance = distances[i] if i < len(distances) else 1.0
        score = max(0.0, 1.0 - distance)

        metadata = metadatas[i] if i < len(metadatas) else {}
        content = documents[i] if i < len(documents) else ""

        result = SemanticSearchResult(
            chunk_id=metadata.get("chunk_id", 0),
            content=content,
            score=round(score, 4),
            source_title=metadata.get("title", ""),
            source_url=metadata.get("url", ""),
            report_id=metadata.get("report_id", ""),
        )
        items.append(result)

    logger.info("[SEMANTIC SEARCH] Query completed")
    logger.info(
        f"[SEMANTIC SEARCH] Results returned: {len(items)}"
    )

    return SemanticSearchResponse(
        query=payload.query,
        count=len(items),
        results=items,
    )
