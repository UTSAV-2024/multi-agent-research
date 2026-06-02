from ddgs import DDGS

from app.config.settings import settings

from app.utils.logger import logger


def search_agent(
    topic,
    num_results=settings.MAX_SEARCH_RESULTS
):

    logger.info(
        f"[SEARCH AGENT] searching for: {topic}"
    )

    try:

        with DDGS() as ddgs:

            results = list(

                ddgs.text(
                    topic,
                    max_results=num_results
                )
            )

    except Exception as e:

        logger.error(
            f"[SEARCH AGENT] error: {e}"
        )

        return []

    sources = []

    for r in results:

        sources.append({

            "title": r.get("title", ""),

            "url": r.get("href", ""),

            "snippet": r.get("body", "")
        })

    logger.info(
        f"[SEARCH AGENT] found "
        f"{len(sources)} sources"
    )

    return sources