import time

from agents.search_agents import search_agent
from agents.content_fetch_agent import content_fetch_agent
from agents.summarizer_agent import summarizer_agent
from agents.factcheck_agent import factcheck_agent
from agents.report_agent import report_agent


def research(topic):

    print(f"\n{'='*60}")
    print("MULTI-AGENT RESEARCH SYSTEM")
    print(f"Topic: {topic}")
    print(f"{'='*60}")

    start = time.time()

    # AGENT 1 — SEARCH
    sources = search_agent(topic)

    if not sources:
        return "No sources found."

    # AGENT 2 — CONTENT FETCH
    enriched_sources = content_fetch_agent(sources)

    # AGENT 3 — SUMMARIZATION
    summaries = summarizer_agent(
        topic,
        enriched_sources
    )

    # AGENT 4 — FACT CHECK
    verified = factcheck_agent(
        topic,
        summaries
    )

    # AGENT 5 — REPORT
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