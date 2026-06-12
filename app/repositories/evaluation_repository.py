# ==========================================
# EVALUATION REPOSITORY
# ==========================================
#
# Persistence layer for evaluation results.
# Failures are logged as warnings and never
# propagate to the caller.
#
# ==========================================

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.db.collections.evaluation_runs_collection import (
    evaluation_runs_collection,
)
from app.utils.logger import logger


# ==========================================
# SAVE EVALUATION
# ==========================================

async def save_evaluation(
    evaluation_type: str,
    request_id: str,
    results: Dict[str, Any],
) -> Optional[str]:
    """Persist an evaluation result to MongoDB.

    Args:
        evaluation_type: Type of evaluation (``"retrieval"`` or ``"evidence"``).
        request_id:      The request ID that triggered this evaluation.
        results:         The evaluation result data to persist.

    Returns:
        The inserted document ID as a string, or ``None`` on failure.
    """
    try:
        document = {
            "evaluation_type": evaluation_type,
            "request_id": request_id,
            "results": results,
            "created_at": datetime.utcnow(),
        }

        result = await evaluation_runs_collection.insert_one(document)

        inserted_id = str(result.inserted_id)
        logger.info(
            "[EVAL REPO] Saved %s evaluation | id=%s | request_id=%s",
            evaluation_type,
            inserted_id,
            request_id,
        )
        return inserted_id

    except Exception as e:
        logger.warning(
            "[EVAL REPO] Failed to save %s evaluation: %s",
            evaluation_type,
            e,
        )
        return None


# ==========================================
# GET RECENT EVALUATIONS
# ==========================================

async def get_recent_evaluations(
    limit: int = 20,
    skip: int = 0,
    evaluation_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve recent evaluation runs, ordered by creation time descending.

    Args:
        limit:           Maximum number of results to return (default 20).
        skip:            Number of results to skip (for pagination).
        evaluation_type: Optional filter by evaluation type
                        (``"retrieval"`` or ``"evidence"``).

    Returns:
        A list of evaluation documents with ``_id`` converted to string.
        Returns an empty list on failure (never raises).
    """
    try:
        query: Dict[str, Any] = {}
        if evaluation_type:
            query["evaluation_type"] = evaluation_type

        cursor = (
            evaluation_runs_collection
            .find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )

        evaluations: List[Dict[str, Any]] = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            # Ensure created_at is ISO string for JSON serialization
            if "created_at" in doc and isinstance(doc["created_at"], datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            evaluations.append(doc)

        logger.info(
            "[EVAL REPO] Retrieved %d evaluations (limit=%d, skip=%d, type=%s)",
            len(evaluations),
            limit,
            skip,
            evaluation_type or "all",
        )
        return evaluations

    except Exception as e:
        logger.warning(
            "[EVAL REPO] Failed to retrieve evaluations: %s",
            e,
        )
        return []
