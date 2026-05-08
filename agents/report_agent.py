import os
import json
from utils.json_utils import clean_json_response, safe_json_parse
from services.llm_service import run_agent
def report_agent(topic, summaries, verified_facts):

    print(f"\n[REPORT AGENT] generating report")

    confirmed = "\n".join(
        f"- {f}"
        for f in verified_facts.get("confirmed_facts", [])
    )

    disputed = "\n".join(
        f"- {f}"
        for f in verified_facts.get("disputed_facts", [])
    )

    single = "\n".join(
        f"- {f}"
        for f in verified_facts.get("single_source_facts", [])
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

    report = run_agent(
        "You are a professional research analyst.",
        prompt,
        max_tokens=1500
    )

    print(f"[REPORT AGENT] report complete")

    return report