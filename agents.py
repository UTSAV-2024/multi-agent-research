from groq import Groq
from dotenv import load_dotenv
from ddgs import DDGS
import trafilatura
import os
import json
import time

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────
def clean_json_response(text):

    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "")

    if text.startswith("```"):
        text = text.replace("```", "")

    text = text.strip()

    return text
def safe_json_parse(result, fallback=None):

    try:
        result = clean_json_response(result)
        return json.loads(result)

    except Exception as e:

        print(f"[JSON ERROR] {e}")

        return fallback
def run_agent(system_prompt, user_prompt, max_tokens=1000):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()

# ─────────────────────────────────────────────
# AGENT 1 — SEARCH AGENT
# ─────────────────────────────────────────────
def search_agent(topic, num_results=5):

    print(f"\n[SEARCH AGENT] searching for: {topic}")

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    topic,
                    max_results=num_results
                )
            )

    except Exception as e:
        print(f"[SEARCH AGENT] error: {e}")
        return []

    sources = []

    for r in results:

        sources.append({
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", "")
        })

    print(f"[SEARCH AGENT] found {len(sources)} sources")

    return sources

# ─────────────────────────────────────────────
# AGENT 2 — CONTENT FETCH AGENT
# ─────────────────────────────────────────────
def content_fetch_agent(sources):

    print(f"\n[CONTENT FETCH AGENT] fetching article content")

    enriched_sources = []

    for i, source in enumerate(sources):

        print(f"[CONTENT FETCH AGENT] processing {i+1}/{len(sources)}")

        try:

            downloaded = trafilatura.fetch_url(source['url'])

            if not downloaded:
                continue

            content = trafilatura.extract(downloaded)

            if not content:
                continue

            enriched_sources.append({
                "title": source['title'],
                "url": source['url'],
                "content": content[:5000]
            })

        except Exception as e:

            print(f"[CONTENT FETCH AGENT] failed: {e}")

    print(f"[CONTENT FETCH AGENT] fetched {len(enriched_sources)} articles")

    return enriched_sources

# ─────────────────────────────────────────────
# AGENT 3 — SUMMARIZER AGENT
# ─────────────────────────────────────────────
def summarizer_agent(topic, sources):

    print(f"\n[SUMMARIZER AGENT] extracting key facts")

    summaries = []

    for i, source in enumerate(sources):

        print(f"[SUMMARIZER AGENT] processing source {i+1}/{len(sources)}")

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

Return ONLY valid JSON like this:

[
    "fact 1",
    "fact 2",
    "fact 3"
]

No explanations.
"""

        result = run_agent(
            "You are a precise fact extraction system. Return only valid JSON.",
            prompt,
            max_tokens=500
        )

        try:

            result = clean_json_response(result)

            facts = safe_json_parse(
            result,
            fallback=[]
            )

            summaries.append({
                "source": source['title'],
                "url": source['url'],
                "facts": facts
            })

        except Exception as e:

            print(f"[SUMMARIZER AGENT] parsing failed: {e}")

            summaries.append({
                "source": source['title'],
                "url": source['url'],
                "facts": [
                    source['content'][:300]
                ]
            })

    print(f"[SUMMARIZER AGENT] extracted facts from {len(summaries)} sources")

    return summaries

# ─────────────────────────────────────────────
# AGENT 4 — FACT CHECK AGENT
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# AGENT 5 — REPORT AGENT
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────
def research(topic):

    print(f"\n{'='*60}")
    print("MULTI-AGENT RESEARCH SYSTEM")
    print(f"Topic: {topic}")
    print(f"{'='*60}")

    start = time.time()

    # agent 1
    sources = search_agent(topic)

    if not sources:
        return "No sources found."

    # agent 2
    enriched_sources = content_fetch_agent(sources)

    # agent 3
    summaries = summarizer_agent(
        topic,
        enriched_sources
    )

    # agent 4
    verified = factcheck_agent(
        topic,
        summaries
    )

    # agent 5
    report = report_agent(
        topic,
        summaries,
        verified
    )

    elapsed = round(
        time.time() - start,
        2
    )

    print(f"\n{'='*60}")
    print(f"RESEARCH COMPLETE in {elapsed}s")
    print(f"{'='*60}\n")

    print(report)

    return report

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    topic = input("Enter research topic: ")

    research(topic)