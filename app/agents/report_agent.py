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

    low_confidence = "\n".join(
        f"- {f}"
        for f in verified_facts.get(
            "low_confidence_facts",
            []
        )
    )

    # Extract unique source-title/URL pairs from fact entries
    # (supports both old per-source and new aggregated summary format)
    seen = set()
    source_lines = []

    for s in summaries:
        # If the summary has facts with source attribution, extract from there
        for fact_entry in s.get("facts", []):
            if isinstance(fact_entry, dict):
                src_title = fact_entry.get("source_title", "")
                src_url = fact_entry.get("source_url", "")
                if src_title and (src_title, src_url) not in seen:
                    seen.add((src_title, src_url))
                    source_lines.append(
                        f"- {src_title}: {src_url}"
                    )

        # Also include the top-level source/url (backward compat)
        src = s.get("source", "")
        url = s.get("url", "")
        if src and (src, url) not in seen:
            seen.add((src, url))
            source_lines.append(
                f"- {src}: {url}"
            )

    sources = "\n".join(source_lines) if source_lines else "(no sources available)"

    prompt = f"""
Write a professional research report on:

{topic}

CONFIRMED FACTS:
{confirmed}

DISPUTED FACTS:
{disputed if disputed else "None"}

LOW CONFIDENCE FACTS:
{low_confidence}

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