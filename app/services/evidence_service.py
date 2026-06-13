# ==========================================
# EVIDENCE RETRIEVAL SERVICE
# ==========================================
#
# Thin abstraction over the hybrid retrieval
# service, tailored for evidence collection.
#
# Responsibilities:
#   1. Call retrieve_chunks() for a fact query
#   2. Enforce source diversity (MAX_CHUNKS_PER_SOURCE)
#   3. Return structured evidence with scores
#   4. Graceful degradation on failure
#
# Usage:
#     from app.services.evidence_service import retrieve_evidence
#     evidence = retrieve_evidence("some fact text", top_k=3)
#
# ==========================================

import time
from typing import Any, Dict, List
from urllib.parse import urlparse

from app.services.retrieval_service import retrieve_chunks
from app.config.settings import settings
from app.utils.logger import logger


# ==========================================
# SOURCE DIVERSITY FILTER
# ==========================================


def _enforce_source_diversity(
    items: List[Dict[str, Any]],
    max_per_source: int,
) -> List[Dict[str, Any]]:
    """
    Enforce a maximum number of chunks per unique source URL.

    Operates on a list of dicts that include a "metadata" key with
    a "url" field. Items are processed in their current order
    (already sorted by score then stable_id), so the first N items
    from each source are kept and the rest are discarded.

    Args:
        items: List of result dicts from retrieve_chunks, where
               each dict has a "metadata" sub-dict with a "url" key.
        max_per_source: Maximum number of chunks allowed from any
                        single source URL.

    Returns:
        Filtered list with at most max_per_source items per source.
    """
    if max_per_source <= 0:
        logger.warning(
            "[EVIDENCE] max_per_source <= 0 (%s), returning empty",
            max_per_source,
        )
        return []

    seen: Dict[str, int] = {}
    filtered: List[Dict[str, Any]] = []
    dropped = 0

    for item in items:
        meta = item.get("metadata", {})
        url = meta.get("url", meta.get("source_url", ""))
        if not url:
            # Items without a URL are kept (conservative approach)
            filtered.append(item)
            continue

        # Extract domain for diversity grouping (e.g., "cnn.com")
        # so different articles from the same domain count
        # toward the same per-source limit.
        try:
            domain = urlparse(url).netloc
        except Exception:
            domain = url  # fall back to full URL on parse failure

        count = seen.get(domain, 0)
        if count < max_per_source:
            seen[domain] = count + 1
            filtered.append(item)
        else:
            dropped += 1

    if dropped > 0:
        logger.info(
            "[EVIDENCE] Source diversity filtered %d chunks "
            "(max_per_source=%d, %d unique sources kept, "
            "%d items returned)",
            dropped,
            max_per_source,
            len(seen),
            len(filtered),
        )

    return filtered


# ==========================================
# RETRIEVE EVIDENCE (PUBLIC API)
# ==========================================


def retrieve_evidence(
    fact: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retrieve top evidence chunks for a given fact.

    Flow:
        1. Call retrieve_chunks() for hybrid semantic+keyword search
        2. Enforce source diversity (MAX_CHUNKS_PER_SOURCE)
        3. Return structured evidence dicts with chunk_id, url, score

    Args:
        fact: The fact text to find supporting evidence for.
        top_k: Number of evidence chunks to retrieve (1–20).

    Returns:
        List of evidence dicts, each with:
            chunk_id (int),
            url      (str),
            score    (float),

        Empty list on failure or no results (graceful degradation).
    """
    start_time = time.perf_counter()
    logger.info(
        "[EVIDENCE] retrieve_evidence fact='%s...' top_k=%d",
        fact[:60],
        top_k,
    )

    if not fact or not fact.strip():
        logger.warning("[EVIDENCE] Empty fact text, returning []")
        return []

    try:
        # -----------------------------------------------
        # 1. Hybrid retrieval
        # -----------------------------------------------
        results = retrieve_chunks(query=fact, top_k=top_k)

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not ids:
            logger.info(
                "[EVIDENCE] No chunks returned for fact"
            )
            return []

        # -----------------------------------------------
        # 2. Build structured item list for diversity
        # -----------------------------------------------
        items: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            stable_id = ids[i]
            metadata = metas[i] if i < len(metas) else {}
            distance = dists[i] if i < len(dists) else 1.0
            score = round(max(0.0, 1.0 - distance), 4)

            items.append({
                "stable_id": stable_id,
                "content": docs[i] if i < len(docs) else "",
                "metadata": metadata,
                "score": score,
            })

        # -----------------------------------------------
        # 3. Enforce source diversity
        # -----------------------------------------------
        max_per_source = settings.MAX_CHUNKS_PER_SOURCE
        if max_per_source > 0 and len(items) > 1:
            items = _enforce_source_diversity(items, max_per_source)

        # -----------------------------------------------
        # 4. Build evidence dicts
        # -----------------------------------------------
        evidence: List[Dict[str, Any]] = []
        for item in items:
            meta = item["metadata"]
            stable_id = item["stable_id"]

            # Parse chunk_id from stable_id (report_id:source_id:chunk_id)
            parts = stable_id.split(":")
            chunk_id = (
                int(parts[-1])
                if len(parts) == 3 and parts[-1].isdigit()
                else 0
            )
            url = meta.get("url", "")
            if url:
                evidence.append({
                    "chunk_id": chunk_id,
                    "url": url,
                    "score": item["score"],
                })

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "[EVIDENCE] retrieve_evidence returned %d items "
            "in %.2fms",
            len(evidence),
            elapsed,
        )

        return evidence

    except Exception as e:
        logger.error(
            "[EVIDENCE] retrieve_evidence failed | %s",
            e,
        )
        return []
