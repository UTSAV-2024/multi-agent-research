from app.db.mongodb import database

from app.utils.logger import logger

reports_collection = database["reports"]


async def ensure_reports_indexes():
    """
    Create indexes for the reports collection.

    Indexes:
        - (request_id, 1): Index for lookups by request_id.
        - (created_at, -1): Descending index for history sorting.

    Idempotent: create_index is a no-op if the index already exists.
    """
    await reports_collection.create_index(
        [("request_id", 1)],
        name="request_id",
    )

    await reports_collection.create_index(
        [("created_at", -1)],
        name="created_at_desc",
    )

    logger.info(
        "[INDEXES] reports collection indexes ensured"
    )