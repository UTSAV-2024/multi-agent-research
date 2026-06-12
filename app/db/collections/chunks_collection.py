from app.db.mongodb import database

from app.utils.logger import logger

chunks_collection = database["chunks"]


async def ensure_chunks_indexes():
    """
    Create indexes for the chunks collection.

    Indexes:
        - (report_id, chunk_id): Compound index for report-scoped
          chunk lookups ordered by chunk_id.

    Idempotent: create_index is a no-op if the index already exists.
    """
    await chunks_collection.create_index(
        [("report_id", 1), ("chunk_id", 1)],
        name="report_id_chunk_id",
    )

    logger.info(
        "[INDEXES] chunks collection indexes ensured"
    )
