# ==========================================
# RETRIEVAL EVALUATION — METRICS
# ==========================================
#
# Data models and helpers for capturing
# retrieval quality measurements.
#
# All measurements are derived from the
# public API of retrieve_chunks() — no
# internal state is accessed directly.
#
# Usage:
#     from app.evaluation.metrics import (
#         RetrievalMetrics,
#         ChunkScore,
#         compute_metrics_from_results,
#     )
#
# ==========================================

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


# ==========================================
# FAILED QUERY TRACKING
# ==========================================


@dataclass(frozen=True)
class FailedQuery:
    """Records a query that failed during batch or benchmark evaluation.

    Attributes:
        query:    The query text that caused the failure.
        query_id: Optional benchmark query identifier.
        error:    The error message describing why the query failed.
    """

    query: str
    query_id: str = ""
    error: str = ""


# ==========================================
# DATA MODELS
# ==========================================


@dataclass(frozen=True)
class ChunkScore:
    """Score information for a single retrieved chunk.

    The raw distance is preserved without assuming a particular
    distance metric. The distance_metric field records which
    metric was used (e.g. ``"cosine"``, ``"l2"``, ``"dot_product"``,
    ``"unknown"``).

    Attributes:
        chunk_id:        Stable identifier of the chunk
                         (e.g. ``rep1:src1:3``).
        raw_distance:    Raw distance value returned by the retrieval
                         function. Lower values indicate higher
                         similarity **only** when the distance metric
                         is a similarity-preserving one (e.g.
                         cosine distance).
        distance_metric: Name of the distance metric that produced
                         ``raw_distance`` (``"cosine"``, ``"l2"``,
                         ``"dot_product"``, or ``"unknown"``).
        source_url:      URL of the source document the chunk
                         belongs to, if available.
        content_preview: First 200 characters of the chunk
                         content for inspection.
    """

    chunk_id: str
    raw_distance: float
    distance_metric: str = "unknown"
    source_url: str = ""
    content_preview: str = ""


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregated retrieval quality metrics for a single query.

    All numeric values are derived from the public return of the
    retrieval function.

    .. note::

        ``average_semantic_score`` and ``average_keyword_score``
        are **not available** from the standard ``retrieve_chunks``
        public return (only the combined hybrid score is exposed).
        They are set to ``0.0`` in standard mode and should be
        interpreted as "not measured".

        ``average_hybrid_score`` is computed from raw distances
        **only** when ``distance_metric == "cosine"``, because
        the conversion ``score = 1 - distance`` is only valid for
        cosine distance. For other metrics it defaults to ``0.0``.

    Attributes:
        retrieval_latency_ms:    End-to-end retrieval latency in
                                 milliseconds.
        retrieved_chunk_count:   Number of chunks returned by the
                                 retrieval function.
        distance_metric:         Distance metric used by the
                                 retrieval function (e.g. ``"cosine"``,
                                 ``"l2"``, ``"unknown"``).
        unique_url_count:        Number of unique source URLs among
                                 the returned chunks.
        unique_domain_count:     Number of unique source domains
                                 (extracted via ``urlparse``) among
                                 the returned chunks.
        average_semantic_score:  Average semantic score across all
                                 returned chunks (0.0 if unavailable).
        average_keyword_score:   Average keyword (BM25) score across
                                 all returned chunks (0.0 if
                                 unavailable).
        average_hybrid_score:    Average combined hybrid score.
                                 Computed from ``1.0 - distance``
                                 **only** when ``distance_metric``
                                 is ``\"cosine\"``; otherwise ``0.0``.
        source_diversity_ratio_url:    Ratio of unique URLs to total
                                 chunks (``unique_url_count /
                                 retrieved_chunk_count``). A value
                                 of 1.0 means every URL is unique.
        source_diversity_ratio_domain: Ratio of unique domains to
                                 total chunks (``unique_domain_count
                                 / retrieved_chunk_count``). A value
                                 of 1.0 means every chunk comes from
                                 a different domain.
        detailed_scores:         Whether ``average_semantic_score``
                                 and ``average_keyword_score``
                                 contain real measurements.
    """

    retrieval_latency_ms: float
    retrieved_chunk_count: int
    distance_metric: str = "unknown"
    unique_url_count: int = 0
    unique_domain_count: int = 0
    average_semantic_score: float = 0.0
    average_keyword_score: float = 0.0
    average_hybrid_score: float = 0.0
    source_diversity_ratio_url: float = 0.0
    source_diversity_ratio_domain: float = 0.0
    detailed_scores: bool = False


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation result for a single query.

    Attributes:
        query:               The query text that was evaluated.
        query_id:            Optional benchmark query identifier.
        category:            Optional query category.
        latency_ms:          Measured retrieval latency in
                             milliseconds.
        retrieved_chunk_count: Number of chunks returned.
        unique_url_count:    Number of unique source URLs.
        unique_domain_count: Number of unique source domains.
        top_chunks:          Score details for each returned
                             chunk, ordered by rank.
        metrics:             Aggregated RetrievalMetrics.
    """

    query: str
    query_id: str = ""
    category: str = ""
    latency_ms: float = 0.0
    retrieved_chunk_count: int = 0
    unique_url_count: int = 0
    unique_domain_count: int = 0
    top_chunks: List[ChunkScore] = field(default_factory=list)
    metrics: Optional[RetrievalMetrics] = None


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregated report across multiple query evaluations.

    Attributes:
        total_queries:      Total number of queries attempted
                            (successful + failed).
        successful_queries: Number of queries that completed
                            without error.
        failed_queries:     Number of queries that raised an
                            exception.
        queries_by_category: Count of successful queries per
                             category.
        failure_details:    List of FailedQuery records with
                            details about each failure.
        overall_metrics:    Average RetrievalMetrics across all
                            successful queries.
        per_query_results:  Individual EvaluationResult for each
                            successful query.
    """

    total_queries: int
    successful_queries: int = 0
    failed_queries: int = 0
    queries_by_category: Dict[str, int] = field(default_factory=dict)
    failure_details: List[FailedQuery] = field(default_factory=list)
    overall_metrics: Optional[RetrievalMetrics] = None
    per_query_results: List[EvaluationResult] = field(default_factory=list)


# ==========================================
# SOURCE DIVERSITY HELPERS
# ==========================================


def _extract_domain(url: str) -> str:
    """Extract the registered domain from a URL.

    Args:
        url: A full URL string (e.g. ``"https://news.cnn.com/article"``).

    Returns:
        The netloc/domain portion (e.g. ``"news.cnn.com"``), or the
        original string if parsing fails.
    """
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


def _count_unique_sources(
    metadatas: List[Dict[str, Any]],
) -> tuple[int, int]:
    """Count unique URLs and unique domains from chunk metadata.

    Args:
        metadatas: List of metadata dicts, each expected to have
                   a ``"url"`` or ``"source_url"`` key.

    Returns:
        A tuple of (unique_url_count, unique_domain_count).
    """
    unique_urls: set[str] = set()
    unique_domains: set[str] = set()

    for m in metadatas:
        url = m.get("url", m.get("source_url", ""))
        if url:
            unique_urls.add(url)
            unique_domains.add(_extract_domain(url))

    return len(unique_urls), len(unique_domains)


# ==========================================
# HYBRID SCORE COMPUTATION
# ==========================================


def _compute_average_hybrid_score(
    distances: List[float],
    distance_metric: str,
) -> float:
    """Compute average hybrid score from raw distances.

    The hybrid score is only meaningful for cosine distance where
    ``score = 1.0 - distance``. For other metrics it returns 0.0.

    Args:
        distances:       List of raw distance values.
        distance_metric: Name of the distance metric.

    Returns:
        Average hybrid score (0.0 if metric is not cosine).
    """
    if not distances:
        return 0.0

    if distance_metric == "cosine":
        scores = [
            round(max(0.0, min(1.0, 1.0 - d)), 4)
            for d in distances
        ]
        return round(statistics.mean(scores), 4)

    return 0.0


# ==========================================
# METRIC COMPUTATION HELPERS
# ==========================================


def compute_metrics_from_results(
    latency_ms: float,
    ids: List[str],
    metadatas: List[Dict[str, Any]],
    distances: List[float],
    distance_metric: str = "unknown",
    detailed_scores: bool = False,
) -> RetrievalMetrics:
    """Compute RetrievalMetrics from the public return of a retrieval function.

    Args:
        latency_ms:       Retrieval latency in milliseconds.
        ids:              List of chunk stable IDs (in rank order).
        metadatas:        List of metadata dicts corresponding to
                          each chunk.
        distances:        List of distance values corresponding to
                          each chunk (lower = more similar).
        distance_metric:  The distance metric used (``"cosine"``,
                          ``"l2"``, ``"dot_product"``, ``"unknown"``).
        detailed_scores:  Whether semantic/keyword scores are real
                          measurements or defaulted to 0.0.

    Returns:
        A fully populated RetrievalMetrics instance.
    """
    chunk_count = len(ids)
    if chunk_count == 0:
        return RetrievalMetrics(
            retrieval_latency_ms=latency_ms,
            retrieved_chunk_count=0,
            distance_metric=distance_metric,
            detailed_scores=detailed_scores,
        )

    # --- Hybrid score (cosine only) ---
    average_hybrid = _compute_average_hybrid_score(
        distances, distance_metric
    )

    # --- Source diversity (URL + domain levels) ---
    unique_urls, unique_domains = _count_unique_sources(metadatas)
    diversity_url = (
        round(unique_urls / chunk_count, 4) if chunk_count > 0 else 0.0
    )
    diversity_domain = (
        round(unique_domains / chunk_count, 4) if chunk_count > 0 else 0.0
    )

    return RetrievalMetrics(
        retrieval_latency_ms=round(latency_ms, 2),
        retrieved_chunk_count=chunk_count,
        distance_metric=distance_metric,
        unique_url_count=unique_urls,
        unique_domain_count=unique_domains,
        average_hybrid_score=average_hybrid,
        source_diversity_ratio_url=diversity_url,
        source_diversity_ratio_domain=diversity_domain,
        detailed_scores=detailed_scores,
    )


def aggregate_metrics(
    results: List[EvaluationResult],
) -> RetrievalMetrics:
    """Average RetrievalMetrics across multiple evaluation results.

    Args:
        results: List of EvaluationResult instances to aggregate.

    Returns:
        A single RetrievalMetrics instance with averaged values.
    """
    if not results:
        return RetrievalMetrics(
            retrieval_latency_ms=0.0,
            retrieved_chunk_count=0,
        )

    latencies = [r.latency_ms for r in results]
    chunks = [r.retrieved_chunk_count for r in results]
    urls = [r.unique_url_count for r in results]
    domains = [r.unique_domain_count for r in results]

    # Collect hybrid scores from top_chunks (if available)
    all_hybrid: List[float] = []
    for r in results:
        if r.metrics and r.metrics.distance_metric == "cosine":
            for c in r.top_chunks:
                score = round(max(0.0, min(1.0, 1.0 - c.raw_distance)), 4)
                all_hybrid.append(score)

    avg_hybrid = round(statistics.mean(all_hybrid), 4) if all_hybrid else 0.0

    # Determine majority distance metric
    metric_counts: Dict[str, int] = {}
    for r in results:
        m = r.metrics.distance_metric if r.metrics else "unknown"
        metric_counts[m] = metric_counts.get(m, 0) + 1
    dominant_metric = max(metric_counts, key=metric_counts.get) if metric_counts else "unknown"

    # Diversity ratios
    diversities_url: List[float] = []
    diversities_domain: List[float] = []
    for r in results:
        if r.retrieved_chunk_count > 0:
            diversities_url.append(r.unique_url_count / r.retrieved_chunk_count)
            diversities_domain.append(r.unique_domain_count / r.retrieved_chunk_count)

    avg_diversity_url = (
        round(statistics.mean(diversities_url), 4) if diversities_url else 0.0
    )
    avg_diversity_domain = (
        round(statistics.mean(diversities_domain), 4) if diversities_domain else 0.0
    )

    any_detailed = any(
        r.metrics is not None and r.metrics.detailed_scores
        for r in results
    )

    return RetrievalMetrics(
        retrieval_latency_ms=round(statistics.mean(latencies), 2),
        retrieved_chunk_count=round(statistics.mean(chunks)),
        distance_metric=dominant_metric,
        unique_url_count=round(statistics.mean(urls)),
        unique_domain_count=round(statistics.mean(domains)),
        average_hybrid_score=avg_hybrid,
        source_diversity_ratio_url=avg_diversity_url,
        source_diversity_ratio_domain=avg_diversity_domain,
        detailed_scores=any_detailed,
    )


# ==========================================
# STABILITY METRIC
# ==========================================


def compute_jaccard_similarity(
    set_a: set[str],
    set_b: set[str],
) -> float:
    """Compute Jaccard similarity between two sets of chunk IDs.

    ``J(A, B) = |A ∩ B| / |A ∪ B|``

    Args:
        set_a: First set of chunk identifiers.
        set_b: Second set of chunk identifiers.

    Returns:
        Jaccard similarity in [0, 1]. Returns 0.0 if both sets
        are empty.
    """
    if not set_a and not set_b:
        return 1.0  # both empty = identical
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return round(len(intersection) / len(union), 4)


def compute_stability(
    runs: List[List[str]],
) -> float:
    """Compute retrieval stability across multiple runs.

    Stability is the average Jaccard similarity between consecutive
    runs of the same query. A value close to 1.0 means the retrieval
    is nearly deterministic.

    Args:
        runs: List of chunk ID lists, one per run.

    Returns:
        Average Jaccard similarity between consecutive runs in [0, 1].
        Returns 1.0 for fewer than 2 runs (trivially stable).
    """
    if len(runs) < 2:
        return 1.0

    similarities: List[float] = []
    for i in range(len(runs) - 1):
        sim = compute_jaccard_similarity(
            set(runs[i]), set(runs[i + 1])
        )
        similarities.append(sim)

    return round(statistics.mean(similarities), 4)
