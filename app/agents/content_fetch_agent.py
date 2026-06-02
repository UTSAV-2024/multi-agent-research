import asyncio
import time
import trafilatura

from app.utils.logger import logger


async def fetch_single_source(
    source,
    index,
    total
):

    logger.info(
        f"[CONTENT FETCH AGENT] processing "
        f"{index+1}/{total}"
    )

    start = time.time()

    try:

        downloaded = await asyncio.wait_for(

            asyncio.to_thread(
                trafilatura.fetch_url,
                source['url']
            ),

            timeout=20
        )

        if not downloaded:

            logger.warning(
                f"[CONTENT FETCH AGENT] "
                f"Failed to download: "
                f"{source['url']}"
            )

            return None

        content = await asyncio.wait_for(

            asyncio.to_thread(
                trafilatura.extract,
                downloaded
            ),

            timeout=20
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

            "content": content[:5000],

            "fetch_time": fetch_time
        }

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