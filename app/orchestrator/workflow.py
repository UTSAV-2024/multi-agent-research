# ==========================================
# CHANGES MADE:
# ==========================================
#
# 1. Converted workflow into async architecture
# 2. Added timeout protection
# 3. Added request_id tracing
# 4. Added async-safe orchestration
# 5. Added cancellation handling
# 6. Added improved observability
# 7. Added safer resilience handling
# 8. Added timeout fallback protection
# 9. Added MongoDB persistence layer
# 10. Added chunking pipeline stage between fetch & summarize
#
# ==========================================

import asyncio
import time

from datetime import datetime

from app.repositories.report_repository import save_report

from app.repositories.chunk_repository import save_chunks

from app.services.chunking_service import (
    chunk_document
)

from app.services.embedding_service import (
    get_embedding_service
)

from app.services.vector_store import (
    VectorStore
)

from app.config.settings import settings

from app.agents.search_agents import search_agent

from app.agents.content_fetch_agent import (
    content_fetch_agent_async
)

from app.agents.summarizer_agent import (
    summarizer_agent_async
)

from app.agents.factcheck_agent import (
    factcheck_agent
)

from app.agents.report_agent import (
    report_agent
)

from app.utils.logger import logger


# ==========================================
# MAIN RESEARCH WORKFLOW
# ==========================================

async def research(
    topic,
    request_id=None
):

    logger.info(
        f"[WORKFLOW] Starting pipeline | "
        f"topic={topic} | "
        f"request_id={request_id}"
    )

    pipeline_start = time.time()

    metrics = {}

    # ==========================================
    # SEARCH AGENT
    # ==========================================

    start = time.time()

    logger.info("[SEARCH AGENT] started")

    sources = search_agent(topic)

    metrics["search_time"] = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[SEARCH AGENT] completed in "
        f"{metrics['search_time']}s"
    )

    if not sources:

        logger.warning(
            "[SEARCH AGENT] No sources found"
        )

        return {
            "topic": topic,
            "status": "failed",
            "message": "No sources found",
            "timestamp": str(datetime.utcnow()),
            "request_id": request_id
        }

    # ==========================================
    # CONTENT FETCH AGENT
    # ==========================================

    start = time.time()

    logger.info(
        "[CONTENT FETCH AGENT] started"
    )

    enriched_sources = await asyncio.wait_for(

        content_fetch_agent_async(
            sources
        ),

        timeout=120
    )

    metrics["fetch_time"] = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[CONTENT FETCH AGENT] completed in "
        f"{metrics['fetch_time']}s"
    )

    # ==========================================
    # FETCH FAILURE TRACKING
    # ==========================================

    failed_fetches = len(sources) - len(enriched_sources)

    metrics["failed_fetches"] = failed_fetches

    logger.info(
        f"[WORKFLOW] failed fetches: "
        f"{failed_fetches}"
    )

    # ==========================================
    # CHUNKING PIPELINE STAGE
    # ==========================================

    start = time.time()

    logger.info(
        "[CHUNKING] started"
    )

    total_chunks = 0

    # Collect all per-source chunk lists for embedding stage
    source_chunks_list = []

    for source in enriched_sources:

        chunks = chunk_document(
            source,
            chunk_size=500,
            overlap=100
        )

        if chunks:

            stored = await save_chunks(
                report_id=request_id or "unknown",
                source_url=source["url"],
                chunks=chunks
            )

            if stored:
                total_chunks += stored
                source_chunks_list.append({
                    "source": source,
                    "chunks": chunks,
                })

    metrics["chunking_time"] = round(
        time.time() - start,
        2
    )

    metrics["total_chunks"] = total_chunks

    logger.info(
        f"[CHUNKING] completed in "
        f"{metrics['chunking_time']}s | "
        f"created {total_chunks} chunks"
    )

    # ==========================================
    # EMBEDDING & VECTOR STORE STAGE
    # ==========================================

    start = time.time()

    logger.info(
        "[EMBEDDING] started"
    )

    total_stored_vectors = 0

    if settings.VECTOR_ENABLED and source_chunks_list:

        embedding_service = get_embedding_service()
        vector_store = VectorStore()
        vector_store.initialize_collection()

        for entry in source_chunks_list:

            source = entry["source"]
            chunks = entry["chunks"]

            # Extract just the text for embedding
            texts = [chunk["content"] for chunk in chunks]

            embeddings = embedding_service.embed_texts(texts)

            stored = vector_store.add_documents(
                report_id=request_id or "unknown",
                source_id=source["url"],
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=embedding_service.model_name,
                embedding_dimension=embedding_service.dimension,
            )

            total_stored_vectors += stored

        vector_store.close()

    metrics["embedding_time"] = round(
        time.time() - start,
        2
    )

    metrics["total_vectors"] = total_stored_vectors

    logger.info(
        f"[EMBEDDING] completed in "
        f"{metrics['embedding_time']}s | "
        f"stored {total_stored_vectors} vectors"
    )

    # ==========================================
    # SUMMARIZER AGENT
    # ==========================================

    start = time.time()

    logger.info(
        "[SUMMARIZER AGENT] started"
    )

    try:

        summaries = await asyncio.wait_for(

            summarizer_agent_async(
                topic,
                enriched_sources
            ),

            timeout=120
        )

    except asyncio.TimeoutError:

        logger.error(
            "[SUMMARIZER AGENT] timeout"
        )

        summaries = []

    metrics["summary_time"] = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[SUMMARIZER AGENT] completed in "
        f"{metrics['summary_time']}s"
    )

    # ==========================================
    # SUMMARY FAILURE TRACKING
    # ==========================================

    failed_summaries = len([

        s for s in summaries
        if s.get("failed")

    ])

    successful_summaries = (
        len(summaries) - failed_summaries
    )

    metrics["failed_summaries"] = failed_summaries

    metrics["successful_summaries"] = successful_summaries

    logger.info(
        f"[WORKFLOW] successful summaries: "
        f"{successful_summaries}"
    )

    logger.info(
        f"[WORKFLOW] failed summaries: "
        f"{failed_summaries}"
    )

    # ==========================================
    # FACTCHECK AGENT
    # ==========================================

    start = time.time()

    logger.info(
        "[FACTCHECK AGENT] started"
    )

    try:

        verified = await asyncio.wait_for(

            factcheck_agent(
                topic,
                summaries
            ),

            timeout=60
        )

    except asyncio.TimeoutError:

        logger.error(
            "[FACTCHECK AGENT] timeout"
        )

        verified = {
            "confirmed_facts": [],
            "disputed_facts": [],
            "low_confidence_facts": []
        }

    metrics["factcheck_time"] = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[FACTCHECK AGENT] completed in "
        f"{metrics['factcheck_time']}s"
    )

    # ==========================================
    # REPORT AGENT
    # ==========================================

    start = time.time()

    logger.info(
        "[REPORT AGENT] started"
    )

    try:

        report = await asyncio.wait_for(

            report_agent(
                topic,
                summaries,
                verified
            ),

            timeout=120
        )

    except asyncio.TimeoutError:

        logger.error(
            "[REPORT AGENT] timeout"
        )

        report = (
            "Report generation timed out."
        )

    metrics["report_time"] = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[REPORT AGENT] completed in "
        f"{metrics['report_time']}s"
    )

    # ==========================================
    # FINAL PIPELINE METRICS
    # ==========================================

    total_time = round(
        time.time() - pipeline_start,
        2
    )

    metrics["total_execution_time"] = total_time

    logger.info(
        f"[WORKFLOW] Pipeline completed in "
        f"{total_time}s"
    )

    # ==========================================
    # STATUS EVALUATION
    # ==========================================

    status = "success"

    if (
        failed_fetches > 0 or
        failed_summaries > 0
    ):
        status = "partial_success"

    if successful_summaries == 0:
        status = "failed"

    logger.info(
        f"[WORKFLOW] Final status: {status}"
    )

    # ==========================================
    # SAVE REPORT TO DATABASE
    # ==========================================

    try:

        await save_report({

            "topic": topic,

            "status": status,

            "report": report,

            "source_count": len(sources),

            "successful_summaries":
                successful_summaries,

            "failed_summaries":
                failed_summaries,

            "failed_fetches":
                failed_fetches,

            "timestamp":
                str(datetime.utcnow()),

            "request_id":
                request_id,

            "metrics":
                metrics

        })

        logger.info(
            "[DATABASE] Report saved successfully"
        )

    except Exception as e:

        logger.error(
            f"[DATABASE ERROR] Failed to save report: {e}"
        )

    # ==========================================
    # FINAL RESPONSE
    # ==========================================

    return {

        "topic": topic,

        "status": status,

        "report": report,

        "source_count": len(sources),

        "successful_summaries":
            successful_summaries,

        "failed_summaries":
            failed_summaries,

        "failed_fetches":
            failed_fetches,

        "timestamp":
            str(datetime.utcnow()),

        "request_id":
            request_id,

        "metrics":
            metrics
    }