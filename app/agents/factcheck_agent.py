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

        for fact in s['facts']:

            all_facts.append(
                f"- {fact} (source: {s['source']})"
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
    "single_source_facts": []
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
                "single_source_facts": []
            }
        )

        # ==========================================
        # VALIDATE KEYS: ensure all expected keys exist
        # ==========================================

        if not isinstance(verified, dict):

            logger.warning(
                "[FACT-CHECK AGENT] Response was not a dict, "
                "falling back to single_source_facts"
            )

            raise ValueError("Response not a dict")

    except Exception as e:

        logger.error(
            f"[FACT-CHECK AGENT] failed | {e}"
        )

        # ==========================================
        # FALLBACK: Try to extract any facts from raw response
        # instead of losing everything
        # ==========================================

        logger.info(
            "[FACT-CHECK AGENT] Attempting fact salvage "
            "from raw response"
        )

        salvaged = extract_json(result)

        if salvaged and isinstance(salvaged, dict):

            verified = {
                "confirmed_facts": (
                    salvaged.get("confirmed_facts", [])
                ),
                "disputed_facts": (
                    salvaged.get("disputed_facts", [])
                ),
                "single_source_facts": (
                    salvaged.get("single_source_facts", [])
                )
            }

        elif salvaged and isinstance(salvaged, list):

            # Response was a list of facts — treat as single_source
            verified = {
                "confirmed_facts": [],
                "disputed_facts": [],
                "single_source_facts": (
                    salvaged[:10]
                )
            }

        else:

            # ==========================================
            # LAST RESORT: Extract any bullet points or
            # numbered items from raw text
            # ==========================================

            lines = result.split("\n")

            extracted = []

            for line in lines:

                line = line.strip()

                if line.startswith("-") or line.startswith("*"):

                    cleaned = line.lstrip("-* ").strip()

                    if cleaned and len(cleaned) > 10:
                        extracted.append(cleaned)

                elif line and line[0].isdigit() and "." in line[:4]:

                    cleaned = line.split(".", 1)[1].strip()

                    if cleaned and len(cleaned) > 10:
                        extracted.append(cleaned)

            if extracted:

                logger.info(
                    f"[FACT-CHECK AGENT] Salvaged "
                    f"{len(extracted)} facts from raw text"
                )

                verified = {
                    "confirmed_facts": [],
                    "disputed_facts": [],
                    "single_source_facts": extracted
                }

            else:

                logger.warning(
                    "[FACT-CHECK AGENT] No facts could be "
                    "salvaged from response"
                )

                verified = {
                    "confirmed_facts": [],
                    "disputed_facts": [],
                    "single_source_facts": []
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
        f"{len(verified.get('single_source_facts', []))}"
    )

    return verified