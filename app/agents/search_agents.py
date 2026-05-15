
from ddgs import DDGS
from utils.logger import logger
from app.services.llm_service import run_agent
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

    logger.info("[SEARCH AGENT] searching...")

    return sources