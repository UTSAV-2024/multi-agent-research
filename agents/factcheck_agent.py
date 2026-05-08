from groq import Groq
from dotenv import load_dotenv
from ddgs import DDGS
import trafilatura
import os
import json
from utils.json_utils import clean_json_response, safe_json_parse
from services.llm_service import run_agent
def factcheck_agent(topic, summaries):

    print(f"\n[FACT-CHECK AGENT] cross-referencing facts")

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

    result = run_agent(
        "You are a fact-checking system.",
        prompt,
        max_tokens=800
    )

    try:

        result = clean_json_response(result)

        verified = safe_json_parse(
                    result,
                    fallback={
            "confirmed_facts": [],
            "disputed_facts": [],
            "single_source_facts": []
            }
        )

    except Exception as e:

        print(f"[FACT-CHECK AGENT] parsing failed: {e}")

        verified = {
            "confirmed_facts": [],
            "disputed_facts": [],
            "single_source_facts": []
        }

    print(
        f"[FACT-CHECK AGENT] confirmed: "
        f"{len(verified.get('confirmed_facts', []))}"
    )

    return verified
