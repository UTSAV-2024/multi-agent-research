import asyncio
import time

import httpx
import trafilatura

from app.config.settings import settings
from app.utils.logger import logger


# ==========================================
# SHARED HTTP CLIENT (lazy init)
# ==========================================

_client_instance = None


def _get_http_client() -> httpx.AsyncClient:
    """Return the shared httpx client, creating it on first call."""
    global _client_instance
    if _client_instance is None:
        _client_instance = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=15.0,
                read=20.0,
                write=10.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
            ),
            follow_redirects=True,
        )
        logger.info(
            "[CONTENT FETCH] Initialised httpx client "
            "(max_connections=10, max_keepalive=5)"
        )
    return _client_instance


async def fetch_single_source(
    source,
    index,
    total,
):

    logger.info(
        f"[CONTENT FETCH AGENT] processing "
        f"{index+1}/{total}"
    )

    start = time.time()

    try:

        client = _get_http_client()

        response = await asyncio.wait_for(

            client.get(
                source["url"],
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            ),

            timeout=20,
        )

        response.raise_for_status()

        html = response.text

        if not html:

            logger.warning(
                f"[CONTENT FETCH AGENT] "
                f"Empty response: "
                f"{source['url']}"
            )

            return None

        content = await asyncio.wait_for(

            asyncio.to_thread(
                trafilatura.extract,
                html,
            ),

            timeout=20,
        )

        if not content:

            logger.warning(
                f"[CONTENT FETCH AGENT] "
                f"No content extracted: "
                f"{source['url']}"
            )

            return None

        fetch_time = round(
            time.time() - start,
            2
        )

        logger.info(
            f"[CONTENT FETCH AGENT] completed "
            f"{source['title']} "
            f"in {fetch_time}s"
        )

        return {

            "title": source['title'],

            "url": source['url'],

            "content": content[:settings.MAX_ARTICLE_LENGTH],

            "fetch_time": fetch_time
        }

    except httpx.HTTPStatusError as e:

        logger.error(
            f"[CONTENT FETCH AGENT] HTTP {e.response.status_code} "
            f"for {source['url']}"
        )

        return None

    except httpx.TimeoutException:

        logger.error(
            f"[CONTENT FETCH AGENT] timeout for "
            f"{source['url']}"
        )

        return None

    except asyncio.TimeoutError:

        logger.error(
            f"[CONTENT FETCH AGENT] timeout for "
            f"{source['url']}"
        )

        return None

    except Exception as e:

        logger.error(
            f"[CONTENT FETCH AGENT] failed for "
            f"{source['url']} | {e}"
        )

        return None


async def content_fetch_agent_async(
    sources
):

    logger.info(
        "[CONTENT FETCH AGENT] "
        "fetching article content concurrently"
    )

    start = time.time()

    tasks = [

        fetch_single_source(
            source,
            index,
            len(sources)
        )

        for index, source in enumerate(sources)
    ]

    results = await asyncio.gather(*tasks)

    enriched_sources = [

        result for result in results
        if result is not None
    ]

    total_time = round(
        time.time() - start,
        2
    )

    logger.info(
        f"[CONTENT FETCH AGENT] fetched "
        f"{len(enriched_sources)} articles "
        f"in {total_time}s"
    )

    if enriched_sources:

        fetch_times = [

            source["fetch_time"]
            for source in enriched_sources
        ]

        avg_fetch = round(
            sum(fetch_times) / len(fetch_times),
            2
        )

        slowest_fetch = max(fetch_times)

        logger.info(
            f"[CONTENT FETCH AGENT] "
            f"average fetch time: "
            f"{avg_fetch}s"
        )

        logger.info(
            f"[CONTENT FETCH AGENT] "
            f"slowest fetch time: "
            f"{slowest_fetch}s"
        )

    failed_fetches = (
        len(sources) - len(enriched_sources)
    )

    logger.info(
        f"[CONTENT FETCH AGENT] "
        f"failed fetches: "
        f"{failed_fetches}"
    )

    return enriched_sources


async def close_http_client():
    """Close the shared httpx client (call on shutdown)."""
    global _client_instance
    if _client_instance is not None:
        await _client_instance.aclose()
        _client_instance = None
        logger.info("[CONTENT FETCH] httpx client closed")
