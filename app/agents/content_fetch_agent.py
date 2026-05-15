import trafilatura
import os
import json
from utils.json_utils import clean_json_response, safe_json_parse
from app.services.llm_service import run_agent
from utils import logger
def content_fetch_agent(sources):

    print(f"\n[CONTENT FETCH AGENT] fetching article content")

    enriched_sources = []

    for i, source in enumerate(sources):

        print(f"[CONTENT FETCH AGENT] processing {i+1}/{len(sources)}")

        try:

            downloaded = trafilatura.fetch_url(source['url'])

            if not downloaded:
                logger.warning(f"Failed to download content from {source['url']}")

            content = trafilatura.extract(downloaded)

            if not content:
                if not content:
                    logger.warning(f"No content extracted from {source['url']}")

            enriched_sources.append({
                "title": source['title'],
                "url": source['url'],
                "content": content[:5000]
            })

        except Exception as e:

            print(f"[CONTENT FETCH AGENT] failed: {e}")

    print(f"[CONTENT FETCH AGENT] fetched {len(enriched_sources)} articles")

    return enriched_sources