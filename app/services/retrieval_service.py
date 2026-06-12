# ==========================================
# RETRIEVAL SERVICE
# ==========================================
#
# Hybrid retrieval combining semantic search
# (ChromaDB) with keyword scoring, merge,
# deduplication, and weighted reranking.
#
# Usage:
#     from app.services.retrieval_service import retrieve_chunks
#     results = retrieve_chunks("query text", top_k=5)
#
# ==========================================

import math
import time
from collections import Counter
from typing import Any, Dict, List, Optional

from app.services.vector_store import VectorStore
from app.config.settings import settings
from app.utils.logger import logger


# ==========================================
# KEYWORD SCORING (BM25-LIKE)
# ==========================================


def _tokenize(text: str) -> List[str]:
    """Simple whitespace+punctuation tokenizer."""
    return text.lower().split()


def _compute_keyword_score(
    query_terms: List[str],
    doc_text: str,
    doc_freq: Dict[str, int],
    total_docs: int,
    avg_doc_len: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> float:
    """
    Compute a BM25-like keyword relevance score for a single document.

    BM25 formula: sum over query terms of
        IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    """
    if not query_terms or not doc_text:
        return 0.0

    doc_terms = _tokenize(doc_text)
    doc_len = len(doc_terms)
    if doc_len == 0:
        return 0.0

    tf_counts = Counter(doc_terms)
    score = 0.0

    for term in query_terms:
        if term not in doc_freq:
            continue

        tf = tf_counts.get(term, 0)
        if tf == 0:
            continue

        idf = math.log(
            (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1.0
        )

        score += idf * (tf * (k1 + 1)) / (
            tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
        )

    return score


def _compute_doc_frequencies(
    all_docs: List[str],
) -> tuple:
    """Compute document frequencies and average document length."""
    total_docs = len(all_docs)
    if total_docs == 0:
        return {}, 0.0

    doc_freq: Dict[str, int] = {}
    total_len = 0

    for doc in all_docs:
        terms = set(_tokenize(doc))
        for term in terms:
            doc_freq[term] = doc_freq.get(term, 0) + 1
        total_len += len(_tokenize(doc))

    avg_doc_len = total_len / total_docs
    return doc_freq, avg_doc_len


# ==========================================
# MERGE & RERANK
# ==========================================


def _normalize_scores(
    items: List[Dict[str, Any]],
    score_key: str,
) -> None:
    """Min-max normalise a score field across items to [0, 1]."""
    scores = [item.get(score_key, 0.0) for item in items]
    if not scores:
        return
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-9:
        for item in items:
            item[score_key] = 0.0
        return
    for item in items:
        item[score_key] = (item.get(score_key, 0.0) - min_s) / (max_s - min_s)


def _merge_and_rerank(
    semantic_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Merge semantic and keyword results, deduplicate by stable_id,
    compute weighted reranking scores, and return top_k.

    final_score = semantic_weight * semantic_score + keyword_weight * keyword_score
    """
    semantic_weight = settings.HYBRID_SEMANTIC_WEIGHT
    keyword_weight = settings.HYBRID_KEYWORD_WEIGHT

    # Build lookup by stable_id
    merged: Dict[str, Dict[str, Any]] = {}

    for item in semantic_results:
        sid = item["stable_id"]
        merged[sid] = item

    for item in keyword_results:
        sid = item["stable_id"]
        if sid in merged:
            # Merge keyword score into existing entry
            merged[sid]["keyword_score"] = item.get("keyword_score", 0.0)
        else:
            merged[sid] = item

    items = list(merged.values())

    if not items:
        return []

    # Normalise scores
    _normalize_scores(items, "semantic_score")
    _normalize_scores(items, "keyword_score")

    # Compute final score
    for item in items:
        sem = item.get("semantic_score", 0.0)
        kw = item.get("keyword_score", 0.0)
        item["score"] = round(
            semantic_weight * sem + keyword_weight * kw, 6
        )

    # Sort by final score descending, then by stable_id for determinism
    items.sort(key=lambda x: (-x["score"], x["stable_id"]))

    logger.info(
        f"[RETRIEVAL] Reranked {len(items)} merged chunks "
        f"(sem_weight={semantic_weight}, kw_weight={keyword_weight})"
    )

    return items[:top_k]


# ==========================================
# MAIN RETRIEVAL FUNCTION
# ==========================================


def retrieve_chunks(
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve semantically similar chunks via hybrid retrieval.

    Flow:
        1. Semantic search via ChromaDB (retrieves top_k * multiplier results)
        2. Keyword search across all stored documents (BM25-like scoring)
        3. Merge results by stable_id, deduplicate
        4. Weighted reranking: 0.7 * semantic + 0.3 * keyword
        5. Return top_k results

    Args:
        query: The search query text.
        top_k: Number of results to return (1–20).

    Returns:
        ChromaDB-style result dict with keys:
            ids, distances, metadatas, documents
        (Backward-compatible shape for the semantic search endpoint.)
    """
    start_time = time.time()
    logger.info(f"[RETRIEVAL] Hybrid query='{query[:50]}...' top_k={top_k}")

    multiplier = settings.HYBRID_RETRIEVAL_MULTIPLIER
    semantic_top_k = top_k * multiplier

    vector_store = VectorStore()
    vector_store.initialize_collection()

    try:
        # ----------------------------------------------------------
        # 1. Semantic retrieval
        # ----------------------------------------------------------
        sem_start = time.time()
        semantic_raw = vector_store.query(
            query_text=query,
            top_k=semantic_top_k,
        )
        sem_time = round((time.time() - sem_start) * 1000, 2)

        sem_ids = semantic_raw.get("ids", [[]])[0]
        sem_docs = semantic_raw.get("documents", [[]])[0]
        sem_metas = semantic_raw.get("metadatas", [[]])[0]
        sem_dists = semantic_raw.get("distances", [[]])[0]

        semantic_count = len(sem_ids)

        semantic_results = []
        for i in range(semantic_count):
            distance = sem_dists[i] if i < len(sem_dists) else 1.0
            metadata = sem_metas[i] if i < len(sem_metas) else {}
            semantic_results.append({
                "stable_id": sem_ids[i],
                "semantic_score": max(0.0, 1.0 - distance),
                "content": sem_docs[i] if i < len(sem_docs) else "",
                "metadata": metadata,
            })

        logger.info(
            f"[RETRIEVAL] Semantic search: {semantic_count} results "
            f"in {sem_time}ms (top_k={semantic_top_k})"
        )

        # ----------------------------------------------------------
        # 2. Keyword retrieval
        # ----------------------------------------------------------
        kw_start = time.time()

        # Get all documents from ChromaDB for keyword scoring
        all_data = vector_store._collection.get()
        all_ids = all_data.get("ids", [])
        all_docs = all_data.get("documents", [])
        all_metas = all_data.get("metadatas", [])

        query_terms = _tokenize(query)
        doc_freq, avg_doc_len = _compute_doc_frequencies(all_docs)
        total_docs = len(all_docs)

        keyword_results = []
        for i in range(total_docs):
            kw_score = _compute_keyword_score(
                query_terms,
                all_docs[i],
                doc_freq,
                total_docs,
                avg_doc_len,
            )
            keyword_results.append({
                "stable_id": all_ids[i],
                "keyword_score": kw_score,
                "content": all_docs[i],
                "metadata": all_metas[i] if i < len(all_metas) else {},
            })

        kw_time = round((time.time() - kw_start) * 1000, 2)
        keyword_count = len(keyword_results)

        # Count how many keyword results have a non-zero score
        nonzero_kw = sum(1 for r in keyword_results if r["keyword_score"] > 0)
        logger.info(
            f"[RETRIEVAL] Keyword scoring: {total_docs} docs scored, "
            f"{nonzero_kw} with non-zero score in {kw_time}ms"
        )

        # ----------------------------------------------------------
        # 3. Merge, deduplicate, rerank
        # ----------------------------------------------------------
        merge_start = time.time()
        final_results = _merge_and_rerank(
            semantic_results, keyword_results, top_k
        )
        merge_time = round((time.time() - merge_start) * 1000, 2)
        merged_count = len(final_results)

        # Unique source count
        unique_urls = set()
        for r in final_results:
            meta = r.get("metadata", {})
            url = meta.get("url", meta.get("source_url", ""))
            if url:
                unique_urls.add(url)

        # ----------------------------------------------------------
        # 4. Metrics logging
        # ----------------------------------------------------------
        total_time = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"[RETRIEVAL] Full report: "
            f"retrieval_time_ms={total_time} | "
            f"semantic_results={semantic_count} | "
            f"keyword_results={keyword_count} | "
            f"merged_results={merged_count} | "
            f"unique_sources={len(unique_urls)} | "
            f"final_results={len(final_results)}"
        )

        # ----------------------------------------------------------
        # 5. Build backward-compatible ChromaDB-style return
        # ----------------------------------------------------------
        ids_out: List[str] = []
        docs_out: List[str] = []
        metas_out: List[Dict] = []
        dists_out: List[float] = []

        for item in final_results:
            ids_out.append(item["stable_id"])
            docs_out.append(item["content"])
            metas_out.append(item.get("metadata", {}))
            # Convert final score back to a "distance" for backward compat
            dists_out.append(max(0.0, 1.0 - item["score"]))

        return {
            "ids": [ids_out],
            "documents": [docs_out],
            "metadatas": [metas_out],
            "distances": [dists_out],
        }

    finally:
        vector_store.close()
