# ==========================================
# SOURCE AGGREGATOR SERVICE
# ==========================================
#
# Converts a list of source dicts into a single
# structured context for single-pass fact extraction.
#
# Public API:
#   aggregate_sources(sources)             - build structured text
#   build_fact_extraction_context(sources)  - alias for clarity
#   estimate_context_size(sources)          - character count
#
# ==========================================

from typing import Any, Dict, List

from app.utils.logger import logger


# ==========================================
# AGGREGATE SOURCES
# ==========================================

def aggregate_sources(
    sources: List[Dict[str, Any]]
) -> str:
    """
    Convert a list of source dicts into a single
    structured text block with source markers.

    Each source is formatted as:

        SOURCE <N>
        Title: ...
        URL: ...
        Content: ...

    Args:
        sources: List of dicts with at least
                 "title", "url", and "content" keys.

    Returns:
        A single string with all sources formatted.
    """
    if not sources:
        logger.warning(
            "[AGGREGATOR] No sources to aggregate"
        )
        return ""

    sections = []

    for i, source in enumerate(sources, 1):

        sections.append(f"SOURCE {i}")
        sections.append(
            f"Title: {source.get('title', 'Untitled')}"
        )
        sections.append(
            f"URL: {source.get('url', '')}"
        )
        sections.append(
            f"Content: {source.get('content', '')}"
        )
        sections.append("")  # blank line separator

    result = "\n".join(sections).strip()

    logger.info(
        f"[AGGREGATOR] Sources aggregated: {len(sources)}"
    )

    return result


# ==========================================
# BUILD FACT EXTRACTION CONTEXT
# ==========================================

def build_fact_extraction_context(
    sources: List[Dict[str, Any]]
) -> str:
    """
    Build a structured context for single-pass fact extraction.
    Currently delegates to aggregate_sources(). May be extended
    in the future with additional formatting or metadata.
    """
    return aggregate_sources(sources)


# ==========================================
# ESTIMATE CONTEXT SIZE
# ==========================================

def estimate_context_size(
    sources: List[Dict[str, Any]]
) -> int:
    """
    Estimate the total context size in characters.

    Args:
        sources: List of source dicts.

    Returns:
        Character count of the aggregated context.
    """
    context = aggregate_sources(sources)

    size = len(context)

    logger.info(
        f"[AGGREGATOR] Total context size: {size} chars"
    )

    return size
