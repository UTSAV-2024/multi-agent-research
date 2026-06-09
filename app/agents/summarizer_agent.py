# ==========================================
# SUMMARIZER AGENT - SINGLE-PASS (REFACTORED)
# ==========================================
#
# Architecture change: replaces per-source
# summarization with a single source-aware
# fact extraction pass.
#
# Before: N sources -> N LLM calls -> merge facts
# After:  N sources -> Source Aggregator -> 1 LLM call -> source-attributed facts
#
# ==========================================

import asyncio
import time

from app.utils.json_utils import (
    clean_json_response,
    safe_json_parse,
)

from app.services.llm_service import run_agent
from app.services.source_aggregator import (
    aggregate_sources,
    estimate_context_size,
)
from app.config.settings import settings
from app.utils.logger import logger


# ==========================================
# SOURCE ATTRIBUTION VALIDATION
# ==========================================

_REQUIRED_FACT_KEYS = {"fact", "source_title", "source_url"}


def _validate_fact_entry(entry: dict) -> bool:
    """Validate that a fact entry has all required keys."""
    missing = _REQUIRED_FACT_KEYS - set(entry.keys())
    if missing:
        logger.warning(
            f"[SUMMARIZER] Skipping fact - missing keys: {missing}"
        )
        return False
    if not isinstance(entry.get("fact"), str) or not entry["fact"].strip():
        logger.warning(
            "[SUMMARIZER] Skipping fact - 'fact' field is empty or not a string"
        )
        return False
    if not isinstance(entry.get("source_title"), str):
        logger.warning(
            "[SUMMARIZER] Skipping fact - 'source_title' is not a string"
        )
        return False
    if not isinstance(entry.get("source_url"), str):
        logger.warning(
            "[SUMMARIZER] Skipping fact - 'source_url' is not a string"
        )
        return False
    return True


def _validate_facts(facts: list) -> list:
    """Validate a list of fact entries, filtering out malformed ones."""
    valid = []
    for entry in facts:
        if not isinstance(entry, dict):
            logger.warning(
                "[SUMMARIZER] Skipping fact - not a dict: %s",
                type(entry).__name__
            )
            continue
        if _validate_fact_entry(entry):
            valid.append(entry)
    return valid


# ==========================================
# FACT EXTRACTION PROMPT
# ==========================================

_SYSTEM_PROMPT = (
    "You are a precise fact extraction system. "
    "Return only valid JSON. No markdown. No explanations. No code fences."
)

_USER_PROMPT_TEMPLATE = """Extract the most important information about: {topic}

Sources:

{aggregated_context}

For each source, extract:
1. key facts
2. important events
3. important people
4. statistics
5. controversies

Return ONLY valid JSON in this format:

{{
    "facts": [
        {{
            "fact": "the extracted fact text",
            "source_title": "the title of the source this fact came from",
            "source_url": "the URL of the source this fact came from"
        }}
    ]
}}

No markdown.
No explanations.
No code fences.
"""


# ==========================================
# MAIN ENTRY POINT - ASYNC
# ==========================================


async def summarizer_agent_async(topic: str, sources: list) -> list:
    """
    Single-pass source-aware fact extraction.

    Args:
        topic: The research topic.
        sources: List of source dicts with title, url, content keys.

    Returns:
        A list containing a single summary dict:
            [{
                "source": "aggregated_sources",
                "facts": [{"fact": "...", "source_title": "...", "source_url": "..."}],
                "summary_time": N.N,
                "failed": False
            }]
        This shape is backward-compatible with the workflow.
    """
    logger.info("[SUMMARIZER AGENT] Single-pass extraction started")
    start = time.time()

    # Aggregate sources
    aggregated_context = aggregate_sources(sources)
    context_size = estimate_context_size(sources)
    logger.info(f"[SUMMARIZER AGENT] Context size: {context_size} chars")

    # Single LLM call
    try:
        prompt = _USER_PROMPT_TEMPLATE.format(
            topic=topic,
            aggregated_context=aggregated_context,
        )

        result = await run_agent(
            _SYSTEM_PROMPT,
            prompt,
            max_tokens=settings.SUMMARY_MAX_TOKENS,
        )

        result = clean_json_response(result)
        parsed = safe_json_parse(
            result,
            fallback={"facts": []},
            expected_type="dict",
        )

        raw_facts = parsed.get("facts", [])
        facts = _validate_facts(raw_facts)
        total_time = round(time.time() - start, 2)

        logger.info(f"[SUMMARIZER AGENT] Facts extracted: {len(facts)}")
        logger.info(f"[SUMMARIZER AGENT] completed in {total_time}s")

        return [
            {
                "source": "aggregated_sources",
                "url": "",
                "facts": facts,
                "summary_time": total_time,
                "failed": False,
            }
        ]

    except Exception as e:
        logger.error(f"[SUMMARIZER AGENT] failed | {e}")
        total_time = round(time.time() - start, 2)
        return [
            {
                "source": "aggregated_sources",
                "url": "",
                "facts": [],
                "summary_time": total_time,
                "failed": True,
            }
        ]


# ==========================================
# SYNC WRAPPER (backward compat)
# ==========================================


def summarizer_agent(topic: str, sources: list) -> list:
    """Sync wrapper for summarizer_agent_async."""
    return asyncio.run(summarizer_agent_async(topic, sources))