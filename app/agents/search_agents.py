from urllib.parse import urlparse

from ddgs import DDGS

from app.config.settings import settings

from app.utils.logger import logger


# ==========================================
# SOURCE RANKING TIERS
# ==========================================


def _get_source_tier(url: str) -> int:
    """Assign a quality tier (1=best, 5=worst) to a source URL."""

    try:
        domain = urlparse(url).netloc.lower()
    except Exception:
        return 5

    if domain.startswith("www."):
        domain = domain[4:]

    # --- Tier 1: Government & major institutions ---

    tier1_keywords = ["nobelprize.org", "nih.gov", "who.int", "britannica.com"]
    for kw in tier1_keywords:
        if kw in domain:
            return 1

    if domain.endswith(".gov") or domain.endswith(".mil") or domain.endswith(".int"):
        return 1

    # --- Tier 2: Encyclopedic ---

    if "wikipedia.org" in domain:
        return 2

    # --- Tier 3: Major educational ---

    if domain.endswith(".edu"):
        return 3

    # --- Tier 4: General authoritative ---

    tier4_keywords = [
        "reuters.com", "ap.org", "bbc.com", "bbc.co.uk",
        "npr.org", "nytimes.com", "wsj.com", "theguardian.com",
        "economist.com", "nature.com", "science.org", "arxiv.org",
        "bloomberg.com", "forbes.com", "cnn.com", "washingtonpost.com",
    ]
    for kw in tier4_keywords:
        if kw in domain:
            return 4

    # --- Tier 5: Everything else ---

    return 5


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
                    max_results=num_results * 3
                )
            )

    except Exception as e:

        logger.error(
            f"[SEARCH AGENT] error: {e}"
        )

        return []

    # ----------------------------------------------------------
    # Build source list with tier annotations + domain dedup
    # ----------------------------------------------------------

    sources = []
    seen_domains = set()

    for r in results:

        url = r.get("href", "")
        if not url:
            continue

        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        # Deduplicate by domain (keep first/highest-ranked)
        if domain in seen_domains:
            continue
        seen_domains.add(domain)

        tier = _get_source_tier(url)

        sources.append({
            "title": r.get("title", ""),
            "url": url,
            "snippet": r.get("body", ""),
            "tier": tier,
        })

    # ----------------------------------------------------------
    # Sort by tier (ascending), then by title length as tiebreaker
    # ----------------------------------------------------------

    sources.sort(key=lambda s: (s["tier"], len(s.get("title", ""))))

    # ----------------------------------------------------------
    # Cap low-authority sources (Tier 5) to at most 1
    # ----------------------------------------------------------

    tier5 = [s for s in sources if s["tier"] == 5]
    tier1_4 = [s for s in sources if s["tier"] <= 4]

    capped = tier1_4 + tier5[:1]

    # Strip tier annotation before returning
    for s in capped:
        s.pop("tier", None)

    logger.info(
        f"[SEARCH AGENT] found "
        f"{len(capped)} sources "
        f"(from {len(results)} raw, "
        f"{len(tier5) - min(1, len(tier5))} low-authority capped)"
    )

    return capped[:num_results]