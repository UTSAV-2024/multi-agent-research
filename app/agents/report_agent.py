import json
import time

from app.services.llm_service import run_agent

from app.config.settings import settings

from app.utils.logger import logger


# ==========================================
# ASYNC REPORT AGENT
# ==========================================

async def report_agent(
    topic,
    summaries,
    verified_facts
):

    logger.info(
        "[REPORT AGENT] generating report"
    )

    start = time.time()

    confirmed = "\n".join(
        f"- {f}"
        for f in verified_facts.get(
            "confirmed_facts",
            []
        )
    )

    disputed = "\n".join(
        f"- {f}"
        for f in verified_facts.get(
            "disputed_facts",
            []
        )
    )

    single = "\n".join(
        f"- {f}"
        for f in verified_facts.get(
            "single_source_facts",
            []
        )
    )

    sources = "\n".join(
        f"- {s['source']}: {s['url']}"
        for s in summaries
    )

    prompt = f"""
Write a professional research report on:

{topic}

CONFIRMED FACTS:
{confirmed}

DISPUTED FACTS:
{disputed if disputed else "None"}

LOW CONFIDENCE FACTS:
{single}

SOURCES:
{sources}

Structure:
1. Executive Summary
2. Key Findings
3. Areas of Uncertainty
4. Sources

Be factual and concise.
"""

    report = await run_agent(

        "You are a professional research analyst.",

        prompt,

        max_tokens=settings.REPORT_MAX_TOKENS
    )

    duration = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[REPORT AGENT] completed in {duration}s"
    )

    return report