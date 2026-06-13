# ==========================================
# RETRIEVAL EVALUATION — EVALUATOR
# ==========================================
#
# Orchestrates retrieval quality measurement
# without modifying the retrieval pipeline.
#
# Uses dependency injection for the retrieval
# function, making it fully testable.
#
# Usage:
#     from app.evaluation.retrieval_evaluator import RetrievalEvaluator
#     from app.evaluation.datasets import DEFAULT_BENCHMARK_QUERIES
#
#     # Default: uses app.services.retrieval_service.retrieve_chunks
#     evaluator = RetrievalEvaluator()
#
#     # Single query
#     result = evaluator.evaluate_query("What is quantum entanglement?")
#
#     # Full benchmark
#     report = evaluator.evaluate_benchmark()
#
#     # Stability (5 runs, average Jaccard)
#     stability = evaluator.evaluate_stability("climate change", runs=5)
#
# ==========================================

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from app.evaluation.datasets import (
    DEFAULT_BENCHMARK_QUERIES,
    BenchmarkQuery,
)
from app.evaluation.metrics import (
    BenchmarkReport,
    ChunkScore,
    EvaluationResult,
    FailedQuery,
    RetrievalMetrics,
    aggregate_metrics,
    compute_jaccard_similarity,
    compute_metrics_from_results,
)
from app.services.retrieval_service import retrieve_chunks
from app.utils.logger import logger


# ==========================================
# DISTANCE METRIC DETECTION
# ==========================================


def _detect_distance_metric(
    retrieval_fn: Callable[..., Dict[str, Any]],
) -> str:
    """Detect the distance metric used by a retrieval function.

    This is a best-effort detection. Currently returns ``"cosine"``
    for the default ``retrieve_chunks`` function (which wraps
    ChromaDB's default cosine distance), and ``"unknown"`` for
    custom functions unless configured via the evaluator.

    Returns:
        ``"cosine"`` or ``"unknown"``.
    """
    if retrieval_fn is retrieve_chunks:
        return "cosine"
    return "unknown"


# ==========================================
# RETRIEVAL EVALUATOR
# ==========================================


class RetrievalEvaluator:
    """Evaluates retrieval quality using an injected retrieval function.

    The evaluator calls the retrieval function, measures observable
    metrics (latency, counts, diversity, scores), and returns
    structured results. It **never** modifies the retrieval
    pipeline.

    Attributes:
        retrieval_fn:   Callable[str, int] -> Dict — defaults to
                        ``retrieve_chunks`` from the retrieval service.
        distance_metric: Distance metric used by the retrieval
                        function (``"cosine"``, ``"l2"``,
                        ``"dot_product"``, or ``"unknown"``).
        default_top_k:  Default number of results to request per query.
    """

    def __init__(
        self,
        retrieval_fn: Optional[
            Callable[..., Dict[str, Any]]
        ] = None,
        distance_metric: Optional[str] = None,
        default_top_k: int = 5,
    ):
        """Initialise the evaluator.

        Args:
            retrieval_fn: Retrieval function to evaluate. Must
                accept ``query`` (str) and ``top_k`` (int) keyword
                arguments and return a ChromaDB-style dict with
                keys ``ids``, ``distances``, ``metadatas``,
                ``documents``.
                Defaults to ``retrieve_chunks``.
            distance_metric: Override the detected distance metric.
                If not provided, it is auto-detected from the
                retrieval function.
            default_top_k: Default number of results when not
                overridden per-query. Must be between 1 and 20.
        """
        self.retrieval_fn = retrieval_fn or retrieve_chunks
        self.distance_metric = (
            distance_metric
            if distance_metric is not None
            else _detect_distance_metric(self.retrieval_fn)
        )
        self.default_top_k = max(1, min(20, default_top_k))

    # ---------------------------------------------------------------
    # SINGLE-QUERY EVALUATION
    # ---------------------------------------------------------------

    def evaluate_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        query_id: str = "",
        category: str = "",
    ) -> EvaluationResult:
        """Evaluate a single query against the retrieval function.

        Args:
            query:     The search query text.
            top_k:     Number of results to request. Falls back to
                       ``default_top_k`` if not provided.
            query_id:  Optional identifier (e.g. ``"hist-001"``).
            category:  Optional category label (e.g. ``"history"``).

        Returns:
            An EvaluationResult with all observable metrics and
            per-chunk score details.

        Raises:
            ValueError: If ``query`` is empty or whitespace-only.
        """
        self._validate_query(query)

        k = top_k if top_k is not None else self.default_top_k
        logger.info(
            "[EVAL] evaluate_query id=%s query='%s...' top_k=%d",
            query_id or "?",
            query[:60],
            k,
        )

        start_time = time.perf_counter()

        try:
            result = self.retrieval_fn(query=query, top_k=k)
        except Exception as e:
            logger.error(
                "[EVAL] Retrieval failed for query '%s...': %s",
                query[:60],
                e,
            )
            return EvaluationResult(
                query=query,
                query_id=query_id,
                category=category,
                latency_ms=0.0,
                retrieved_chunk_count=0,
                top_chunks=[],
                metrics=RetrievalMetrics(
                    retrieval_latency_ms=0.0,
                    retrieved_chunk_count=0,
                    distance_metric=self.distance_metric,
                ),
            )

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        # --- Parse the ChromaDB-style return ---
        ids: List[str] = result.get("ids", [[]])[0]
        distances: List[float] = result.get(
            "distances", [[]]
        )[0]
        metadatas: List[Dict[str, Any]] = result.get(
            "metadatas", [[]]
        )[0]
        documents: List[str] = result.get(
            "documents", [[]]
        )[0]

        # --- Build per-chunk scores ---
        top_chunks: List[ChunkScore] = []
        for i in range(len(ids)):
            chunk_id = ids[i]
            dist = distances[i] if i < len(distances) else 1.0
            meta = metadatas[i] if i < len(metadatas) else {}
            url = meta.get("url", meta.get("source_url", ""))
            content = (
                documents[i][:200] if i < len(documents) else ""
            )

            top_chunks.append(
                ChunkScore(
                    chunk_id=chunk_id,
                    raw_distance=dist,
                    distance_metric=self.distance_metric,
                    source_url=url,
                    content_preview=content,
                )
            )

        # --- Count unique domains (for flat fields on result) ---
        unique_urls: set[str] = set()
        unique_domains: set[str] = set()
        for m in metadatas:
            u = m.get("url", m.get("source_url", ""))
            if u:
                unique_urls.add(u)
                try:
                    unique_domains.add(urlparse(u).netloc or u)
                except Exception:
                    unique_domains.add(u)

        # --- Compute aggregated metrics ---
        metrics = compute_metrics_from_results(
            latency_ms=elapsed_ms,
            ids=ids,
            metadatas=metadatas,
            distances=distances,
            distance_metric=self.distance_metric,
        )

        logger.info(
            "[EVAL] Result: %d chunks, "
            "%d urls, %d domains, "
            "%.2fms, avg_hybrid=%.4f",
            metrics.retrieved_chunk_count,
            metrics.unique_url_count,
            metrics.unique_domain_count,
            metrics.retrieval_latency_ms,
            metrics.average_hybrid_score,
        )

        return EvaluationResult(
            query=query,
            query_id=query_id,
            category=category,
            latency_ms=elapsed_ms,
            retrieved_chunk_count=len(ids),
            unique_url_count=len(unique_urls),
            unique_domain_count=len(unique_domains),
            top_chunks=top_chunks,
            metrics=metrics,
        )

    # ---------------------------------------------------------------
    # BATCH EVALUATION
    # ---------------------------------------------------------------

    def evaluate_queries(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> tuple[List[EvaluationResult], List[FailedQuery]]:
        """Evaluate a list of query strings.

        Args:
            queries: List of query texts to evaluate.
            top_k:   Number of results per query. Falls back to
                     ``default_top_k`` if not provided.

        Returns:
            A tuple of ``(results, failures)`` where ``results``
            contains successful evaluations and ``failures``
            contains details about queries that raised
            ``ValueError`` (empty/whitespace queries).

            Invalid queries are gracefully skipped without
            terminating the batch.
        """
        logger.info(
            "[EVAL] Batch evaluate: %d queries, top_k=%s",
            len(queries),
            top_k or self.default_top_k,
        )

        results: List[EvaluationResult] = []
        failures: List[FailedQuery] = []

        for i, query in enumerate(queries, 1):
            logger.info(
                "[EVAL] Batch query %d/%d", i, len(queries)
            )
            try:
                result = self.evaluate_query(
                    query=query, top_k=top_k
                )
                results.append(result)
            except ValueError as e:
                logger.warning(
                    "[EVAL] Batch query %d/%d skipped: %s",
                    i,
                    len(queries),
                    e,
                )
                failures.append(
                    FailedQuery(
                        query=query,
                        error=str(e),
                    )
                )

        logger.info(
            "[EVAL] Batch complete: %d succeeded, %d failed",
            len(results),
            len(failures),
        )

        return results, failures

    # ---------------------------------------------------------------
    # BENCHMARK EVALUATION
    # ---------------------------------------------------------------

    def evaluate_benchmark(
        self,
        queries: Optional[List[BenchmarkQuery]] = None,
        top_k: Optional[int] = None,
    ) -> BenchmarkReport:
        """Run the full benchmark suite.

        Args:
            queries: List of BenchmarkQuery entries to evaluate.
                     Defaults to ``DEFAULT_BENCHMARK_QUERIES``.
            top_k:   Number of results per query. Falls back to
                     ``default_top_k``.

        Returns:
            A BenchmarkReport with per-query results, aggregated
            metrics, failure details, and category breakdowns.
            Invalid queries are skipped gracefully.
        """
        entries = (
            queries
            if queries is not None
            else DEFAULT_BENCHMARK_QUERIES
        )
        k = top_k if top_k is not None else self.default_top_k

        logger.info(
            "[EVAL] Benchmark: %d queries, top_k=%d",
            len(entries),
            k,
        )

        per_query_results: List[EvaluationResult] = []
        failure_details: List[FailedQuery] = []
        queries_by_category: Dict[str, int] = {}

        for entry in entries:
            try:
                result = self.evaluate_query(
                    query=entry.query,
                    top_k=k,
                    query_id=entry.id,
                    category=entry.category,
                )
                per_query_results.append(result)
                queries_by_category[entry.category] = (
                    queries_by_category.get(entry.category, 0) + 1
                )
            except ValueError as e:
                logger.warning(
                    "[EVAL] Benchmark query %s skipped: %s",
                    entry.id,
                    e,
                )
                failure_details.append(
                    FailedQuery(
                        query=entry.query,
                        query_id=entry.id,
                        error=str(e),
                    )
                )

        # Aggregate metrics across successful queries only
        overall = aggregate_metrics(per_query_results) if per_query_results else None

        logger.info(
            "[EVAL] Benchmark complete: %d passed, %d failed",
            len(per_query_results),
            len(failure_details),
        )

        return BenchmarkReport(
            total_queries=len(entries),
            successful_queries=len(per_query_results),
            failed_queries=len(failure_details),
            queries_by_category=queries_by_category,
            failure_details=failure_details,
            overall_metrics=overall,
            per_query_results=per_query_results,
        )

    # ---------------------------------------------------------------
    # STABILITY EVALUATION
    # ---------------------------------------------------------------

    def evaluate_stability(
        self,
        query: str,
        runs: int = 5,
        top_k: Optional[int] = None,
    ) -> float:
        """Measure retrieval stability (determinism) across multiple runs.

        Runs the same query ``runs`` times and computes the average
        Jaccard similarity between consecutive runs. A value close
        to 1.0 indicates highly deterministic retrieval.

        Args:
            query: The search query text.
            runs:  Number of times to run the query (default 5).
            top_k: Number of results to request per run. Falls back
                   to ``default_top_k``.

        Returns:
            Average Jaccard similarity between consecutive runs
            in [0, 1]. Returns 1.0 for fewer than 2 runs.
        """
        self._validate_query(query)

        k = top_k if top_k is not None else self.default_top_k
        logger.info(
            "[EVAL] evaluate_stability query='%s...' "
            "runs=%d top_k=%d",
            query[:60],
            runs,
            k,
        )

        all_ids: List[List[str]] = []
        for run_num in range(1, runs + 1):
            try:
                result = self.retrieval_fn(query=query, top_k=k)
                ids = result.get("ids", [[]])[0]
                all_ids.append(ids)
            except Exception as e:
                logger.warning(
                    "[EVAL] Stability run %d/%d failed: %s",
                    run_num,
                    runs,
                    e,
                )
                all_ids.append([])

        # Compute average Jaccard between consecutive runs
        if len(all_ids) < 2:
            return 1.0

        similarities: List[float] = []
        for i in range(len(all_ids) - 1):
            sim = compute_jaccard_similarity(
                set(all_ids[i]), set(all_ids[i + 1])
            )
            similarities.append(sim)

        avg_stability = round(
            statistics.mean(similarities), 4
        )
        logger.info(
            "[EVAL] Stability: %f (%d runs, top_k=%d)",
            avg_stability,
            runs,
            k,
        )

        return avg_stability

    # ---------------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> None:
        """Validate that a query is non-empty.

        Args:
            query: The query string to validate.

        Raises:
            ValueError: If query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError(
                "Query must be a non-empty string"
            )
