import time

from app.utils.json_utils import (
    clean_json_response,
    safe_json_parse,
    extract_json
)

from app.services.llm_service import run_agent
from app.utils.logger import logger


# ==========================================
# ASYNC FACT-CHECK AGENT
# ==========================================

async def factcheck_agent(
    topic,
    summaries
):

    logger.info(
        "[FACT-CHECK AGENT] cross-referencing facts"
    )

    start = time.time()

    all_facts = []

    for s in summaries:

        for fact_entry in s['facts']:

            # Handle both string facts (old format) and dict facts (new format)
            if isinstance(fact_entry, dict):
                fact_text = fact_entry.get(
                    "fact",
                    str(fact_entry)
                )
                source_label = fact_entry.get(
                    "source_title",
                    s.get("source", "unknown")
                )
            else:
                fact_text = str(fact_entry)
                source_label = s.get("source", "unknown")

            all_facts.append(
                f"- {fact_text} (source: {source_label})"
            )

    facts_text = "\n".join(all_facts)

    prompt = f"""
Topic: {topic}

Facts gathered from multiple sources:

{facts_text}

Tasks:
1. identify confirmed facts
2. identify contradictions
3. identify weak/single-source claims

Return ONLY valid JSON:

{{
    "confirmed_facts": [],
    "disputed_facts": [],
    "low_confidence_facts": []
}}
"""

    try:

        result = await run_agent(
            "You are a fact-checking system.",
            prompt,
            max_tokens=800
        )

        result = clean_json_response(result)

        verified = safe_json_parse(
            result,
            fallback={
                "confirmed_facts": [],
                "disputed_facts": [],
                "low_confidence_facts": []
            },
            expected_type="dict"
        )

        # ==========================================
        # DIAGNOSTIC: Log parsed structure and counts
        # ==========================================

        logger.info(
            "[FACTCHECK DEBUG] parsed keys=%s",
            list(verified.keys()) if isinstance(verified, dict) else type(verified).__name__
        )

        logger.info(
            "[FACTCHECK DEBUG] confirmed=%s disputed=%s low=%s",
            len(verified.get("confirmed_facts", [])),
            len(verified.get("disputed_facts", [])),
            len(verified.get("low_confidence_facts", [])),
        )

        if isinstance(verified, dict) and "facts" in verified:
            logger.warning(
                "[FACTCHECK DEBUG] recovered data stored under 'facts' key (%s items)",
                len(verified["facts"])
            )

    except Exception as e:

        logger.error(
            f"[FACT-CHECK AGENT] failed | {e}"
        )

        verified = {

            "confirmed_facts": [],
            "disputed_facts": [],
            "low_confidence_facts": []
        }

    duration = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[FACT-CHECK AGENT] completed in {duration}s"
    )

    logger.info(
        f"[FACT-CHECK AGENT] confirmed facts: "
        f"{len(verified.get('confirmed_facts', []))}"
    )

    logger.info(
        f"[FACT-CHECK AGENT] disputed facts: "
        f"{len(verified.get('disputed_facts', []))}"
    )

    logger.info(
        f"[FACT-CHECK AGENT] low-confidence facts: "
        f"{len(verified.get('low_confidence_facts', []))}"
    )

    return verified