from datetime import datetime

from app.db.collections.chunks_collection import (
    chunks_collection
)

from app.utils.logger import logger


async def save_chunks(
    report_id: str,
    source_url: str,
    chunks: list
):

    """
    Store chunked content for a source.

    Args:
        report_id: The parent report ID.
        source_url: The source URL the chunks belong to.
        chunks: List of chunks from chunking_service.
    """

    if not chunks:

        return

    chunk_docs = []

    for i, chunk in enumerate(chunks):

        chunk_docs.append({

            "report_id": report_id,

            "source_url": source_url,

            "chunk_index": i,

            "content": chunk["content"],

            "created_at": datetime.utcnow()

        })

    result = await chunks_collection.insert_many(
        chunk_docs
    )

    logger.info(
        f"[CHUNK REPO] Stored "
        f"{len(chunk_docs)} chunks "
        f"for report_id={report_id}"
    )

    return len(chunk_docs)


async def get_chunks_by_report_id(
    report_id: str
):

    cursor = (
        chunks_collection
        .find({"report_id": report_id})
        .sort("chunk_index", 1)
    )

    chunks = []

    async for chunk in cursor:

        chunk["_id"] = str(chunk["_id"])

        chunks.append(chunk)

    return chunks


async def delete_chunks_by_report_id(
    report_id: str
):

    result = await chunks_collection.delete_many(
        {"report_id": report_id}
    )

    logger.info(
        f"[CHUNK REPO] Deleted "
        f"{result.deleted_count} chunks "
        f"for report_id={report_id}"
    )

    return result.deleted_count
