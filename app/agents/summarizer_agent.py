import asyncio
import time

from app.utils.json_utils import (
    clean_json_response,
    safe_json_parse
)

from app.services.llm_service import run_agent

from app.config.settings import settings

from app.utils.logger import logger


# ==========================================
# NEW:
# Async function for summarizing ONE source
# ==========================================

async def summarize_single_source(
    topic,
    source,
    index,
    total
):

    logger.info(
        f"[SUMMARIZER AGENT] processing source {index+1}/{total}"
    )

    start = time.time()

    prompt = f"""
Extract the most important information about: {topic}

Focus on:
1. key claims
2. dates/events
3. important people
4. controversies
5. statistics

Source title:
{source['title']}

Source content:
{source['content']}

Return ONLY valid JSON in this format:

{{
    "facts": [
        "fact 1",
        "fact 2",
        "fact 3"
    ]
}}

No markdown.
No explanations.
No code blocks.
"""

    try:

        result = await run_agent(
            
            "You are a precise fact extraction system. Return only valid JSON.",

            prompt,

            settings.SUMMARY_MAX_TOKENS
        )

        # ==========================================
        # Better JSON cleanup pipeline
        # ==========================================

        result = clean_json_response(result)

        parsed = safe_json_parse(
            result,
            fallback={"facts": []}
        )

        facts = parsed.get("facts", [])

        summary_time = round(
            time.time() - start,
            2
        )

        logger.info(
            f"[SUMMARIZER AGENT] completed {source['title']} in {summary_time}s"
        )

        return {
            "source": source['title'],
            "url": source['url'],
            "facts": facts,
            "summary_time": summary_time
        }

    except Exception as e:

        logger.error(
            f"[SUMMARIZER AGENT] failed for {source['title']} | {e}"
        )

        return {
            "source": source['title'],
            "url": source['url'],
            "facts": [
                source['content'][:300]
            ],
            "summary_time": 0,
            "failed": True
        }


# ==========================================
# Concurrent summarization orchestrator
# ==========================================

async def summarizer_agent_async(
    topic,
    sources
):

    logger.info(
        "[SUMMARIZER AGENT] extracting key facts concurrently"
    )

    start = time.time()

    # ==========================================
    # Create concurrent summarization tasks
    # ==========================================

    tasks = [

        summarize_single_source(
            topic,
            source,
            index,
            len(sources)
        )

        for index, source in enumerate(sources)
    ]

    # ==========================================
    # Execute ALL summarizations concurrently
    # ==========================================

    summaries = await asyncio.gather(*tasks)

    total_time = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[SUMMARIZER AGENT] completed in {total_time}s"
    )

    # ==========================================
    # Observability metrics
    # ==========================================

    successful = [

        s for s in summaries
        if not s.get("failed")
    ]

    failed = [

        s for s in summaries
        if s.get("failed")
    ]

    logger.info(
        f"[SUMMARIZER AGENT] successful summaries: {len(successful)}"
    )

    logger.info(
        f"[SUMMARIZER AGENT] failed summaries: {len(failed)}"
    )

    if successful:

        summary_times = [

            s["summary_time"]
            for s in successful
        ]

        avg_summary = round(
            sum(summary_times) / len(summary_times),
            2
        )

        slowest_summary = max(summary_times)

        logger.info(
            f"[SUMMARIZER AGENT] average summary time: {avg_summary}s"
        )

        logger.info(
            f"[SUMMARIZER AGENT] slowest summary time: {slowest_summary}s"
        )

    return summaries


# ==========================================
# Main sync wrapper
# ==========================================

def summarizer_agent(
    topic,
    sources
):

    return asyncio.run(
        summarizer_agent_async(
            topic,
            sources
        )
    )