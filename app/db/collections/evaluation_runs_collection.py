# ==========================================
# EVALUATION RUNS COLLECTION
# ==========================================
#
# MongoDB collection for persisting evaluation
# results for future trend analysis.
#
# ==========================================

from app.db.mongodb import database
from app.utils.logger import logger

evaluation_runs_collection = database["evaluation_runs"]


async def ensure_evaluation_runs_indexes():
    """Create indexes for the evaluation_runs collection.

    Indexes:
        - (created_at, -1): Descending index for history queries.
        - (evaluation_type, 1): Index for filtering by type.
        - (request_id, 1): Index for deduplication lookups.

    Idempotent: create_index is a no-op if the index already exists.
    """
    await evaluation_runs_collection.create_index(
        [("created_at", -1)],
        name="created_at_desc",
    )

    await evaluation_runs_collection.create_index(
        [("evaluation_type", 1)],
        name="evaluation_type",
    )

    await evaluation_runs_collection.create_index(
        [("request_id", 1)],
        name="request_id",
    )

    logger.info(
        "[INDEXES] evaluation_runs collection indexes ensured"
    )
