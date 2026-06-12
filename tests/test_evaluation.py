# ==========================================
# RETRIEVAL EVALUATION FRAMEWORK — TESTS
# ==========================================
#
# Tests for datasets, metrics, and the
# RetrievalEvaluator.
#
# Run with:
#     pytest tests/test_evaluation.py -v
#
# ==========================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock

from app.evaluation.datasets import (
    DEFAULT_BENCHMARK_QUERIES,
    BENCHMARK_FAILURE_QUERIES,
    BenchmarkQuery,
)
from app.evaluation.metrics import (
    RetrievalMetrics,
    ChunkScore,
    EvaluationResult,
    BenchmarkReport,
    FailedQuery,
    compute_metrics_from_results,
    aggregate_metrics,
    compute_jaccard_similarity,
    compute_stability,
    _extract_domain,
    _count_unique_sources,
    _compute_average_hybrid_score,
)


# ==========================================
# DATASETS TESTS
# ==========================================


class TestBenchmarkQuery:
    """Tests for the BenchmarkQuery dataclass."""

    def test_frozen_dataclass(self):
        """BenchmarkQuery instances are immutable."""
        q = BenchmarkQuery(
            id="test-001",
            query="test query",
            category="test",
        )
        with pytest.raises(AttributeError):
            q.query = "changed"  # type: ignore[misc]

    def test_default_notes_is_empty(self):
        """notes field defaults to empty string."""
        q = BenchmarkQuery(
            id="test-001", query="test", category="test"
        )
        assert q.notes == ""


class TestDefaultBenchmarkQueries:
    """Tests for DEFAULT_BENCHMARK_QUERIES."""

    def test_has_35_queries(self):
        """DEFAULT_BENCHMARK_QUERIES contains exactly 35 entries."""
        assert len(DEFAULT_BENCHMARK_QUERIES) == 35

    def test_all_have_unique_ids(self):
        """Every query has a unique id."""
        ids = [q.id for q in DEFAULT_BENCHMARK_QUERIES]
        assert len(ids) == len(set(ids))

    def test_all_have_non_empty_queries(self):
        """Every query has non-empty query text."""
        for q in DEFAULT_BENCHMARK_QUERIES:
            assert q.query.strip(), f"Empty query for {q.id}"

    def test_queries_cover_all_categories(self):
        """Queries cover 6 named categories."""
        categories = {q.category for q in DEFAULT_BENCHMARK_QUERIES}
        assert len(categories) >= 6

    def test_each_category_has_at_least_five_queries(self):
        """Every named category has at least 5 queries."""
        from collections import Counter

        counts = Counter(
            q.category for q in DEFAULT_BENCHMARK_QUERIES
        )
        for cat, count in counts.items():
            assert count >= 5, (
                f"Category '{cat}' has only {count} queries"
            )

    def test_no_failure_queries_in_default(self):
        """DEFAULT_BENCHMARK_QUERIES contains no failure queries."""
        for q in DEFAULT_BENCHMARK_QUERIES:
            assert q.category != "failure"


class TestBenchmarkFailureQueries:
    """Tests for BENCHMARK_FAILURE_QUERIES."""

    def test_has_5_queries(self):
        """BENCHMARK_FAILURE_QUERIES contains exactly 5 entries."""
        assert len(BENCHMARK_FAILURE_QUERIES) == 5

    def test_all_are_failure_category(self):
        """All failure queries have category='failure'."""
        for q in BENCHMARK_FAILURE_QUERIES:
            assert q.category == "failure"

    def test_includes_empty_and_gibberish(self):
        """Failure queries include empty, whitespace, and gibberish."""
        queries = [q.query for q in BENCHMARK_FAILURE_QUERIES]
        assert "" in queries
        assert "      " in queries
        assert "asdfghjkl" in queries


# ==========================================
# HELPER TESTS
# ==========================================


class TestExtractDomain:
    """Tests for _extract_domain()."""

    def test_standard_url(self):
        """Standard URL extracts netloc."""
        assert _extract_domain("https://cnn.com/article") == "cnn.com"

    def test_url_with_www(self):
        """URL with www preserves full netloc."""
        assert (
            _extract_domain("https://www.cnn.com/article")
            == "www.cnn.com"
        )

    def test_url_with_subdomain(self):
        """URL with subdomain preserves full netloc."""
        assert (
            _extract_domain("https://news.bbc.co.uk/story")
            == "news.bbc.co.uk"
        )

    def test_empty_url(self):
        """Empty URL returns empty string."""
        assert _extract_domain("") == ""

    def test_invalid_url(self):
        """Invalid URL returns the string itself."""
        result = _extract_domain("not-a-url")
        assert result == "not-a-url" or result == ""


class TestCountUniqueSources:
    """Tests for _count_unique_sources()."""

    def test_counts_urls_and_domains(self):
        """Counts unique URLs and unique domains."""
        metadatas = [
            {"url": "https://cnn.com/a"},
            {"url": "https://cnn.com/b"},
            {"url": "https://reuters.com/c"},
        ]
        urls, domains = _count_unique_sources(metadatas)
        assert urls == 3  # 3 unique URLs
        assert domains == 2  # 2 unique domains

    def test_empty_metadata(self):
        """Empty metadata yields zero counts."""
        urls, domains = _count_unique_sources([])
        assert urls == 0
        assert domains == 0

    def test_missing_url(self):
        """Chunks without URLs are not counted."""
        metadatas = [{}, {"url": ""}, {"source_url": ""}]
        urls, domains = _count_unique_sources(metadatas)
        assert urls == 0
        assert domains == 0


class TestComputeAverageHybridScore:
    """Tests for _compute_average_hybrid_score()."""

    def test_cosine_metric(self):
        """Cosine distance is converted to score correctly."""
        avg = _compute_average_hybrid_score(
            [0.1, 0.3, 0.5], "cosine"
        )
        expected = ((1.0 - 0.1) + (1.0 - 0.3) + (1.0 - 0.5)) / 3
        assert avg == pytest.approx(expected, 0.001)

    def test_l2_metric_returns_zero(self):
        """Non-cosine metrics return 0.0."""
        avg = _compute_average_hybrid_score([0.1, 0.2], "l2")
        assert avg == 0.0

    def test_empty_distances(self):
        """Empty list returns 0.0."""
        avg = _compute_average_hybrid_score([], "cosine")
        assert avg == 0.0

    def test_negative_distance_clamped(self):
        """Negative cosine distances are clamped to [0, 1]."""
        avg = _compute_average_hybrid_score([-0.5], "cosine")
        assert avg == 1.0


class TestJaccardSimilarity:
    """Tests for compute_jaccard_similarity()."""

    def test_identical_sets(self):
        """Identical sets yield Jaccard = 1.0."""
        sim = compute_jaccard_similarity(
            {"a", "b", "c"}, {"a", "b", "c"}
        )
        assert sim == 1.0

    def test_disjoint_sets(self):
        """Completely disjoint sets yield Jaccard = 0.0."""
        sim = compute_jaccard_similarity(
            {"a", "b"}, {"c", "d"}
        )
        assert sim == 0.0

    def test_partial_overlap(self):
        """Partially overlapping sets yield 0 < J < 1."""
        sim = compute_jaccard_similarity(
            {"a", "b", "c"}, {"b", "c", "d"}
        )
        # Intersection = {"b", "c"} (2), Union = {"a", "b", "c", "d"} (4)
        assert sim == 0.5

    def test_both_empty(self):
        """Both empty sets yield Jaccard = 1.0 (trivially identical)."""
        sim = compute_jaccard_similarity(set(), set())
        assert sim == 1.0

    def test_one_empty(self):
        """One empty set yields Jaccard = 0.0."""
        sim = compute_jaccard_similarity(
            {"a", "b"}, set()
        )
        assert sim == 0.0


class TestComputeStability:
    """Tests for compute_stability()."""

    def test_identical_runs(self):
        """All runs identical yields stability = 1.0."""
        runs = [
            ["a", "b", "c"],
            ["a", "b", "c"],
            ["a", "b", "c"],
        ]
        stab = compute_stability(runs)
        assert stab == 1.0

    def test_two_runs(self):
        """Two runs with partial overlap."""
        runs = [["a", "b", "c"], ["b", "c", "d"]]
        stab = compute_stability(runs)
        assert stab == 0.5

    def test_single_run_trivially_stable(self):
        """Fewer than 2 runs returns 1.0."""
        stab = compute_stability([["a", "b"]])
        assert stab == 1.0

    def test_empty_runs(self):
        """Empty list returns 1.0."""
        stab = compute_stability([])
        assert stab == 1.0


# ==========================================
# METRICS DATA MODELS TESTS
# ==========================================


class TestChunkScore:
    """Tests for the ChunkScore dataclass."""

    def test_defaults(self):
        """Default values for optional fields."""
        c = ChunkScore(
            chunk_id="r:s:1", raw_distance=0.15
        )
        assert c.chunk_id == "r:s:1"
        assert c.raw_distance == 0.15
        assert c.distance_metric == "unknown"
        assert c.source_url == ""
        assert c.content_preview == ""

    def test_with_cosine_metric(self):
        """ChunkScore works with explicit distance_metric."""
        c = ChunkScore(
            chunk_id="r:s:1",
            raw_distance=0.15,
            distance_metric="cosine",
        )
        assert c.distance_metric == "cosine"


class TestRetrievalMetrics:
    """Tests for the RetrievalMetrics dataclass."""

    def test_defaults(self):
        """Default values are sensible."""
        m = RetrievalMetrics(
            retrieval_latency_ms=42.5,
            retrieved_chunk_count=5,
        )
        assert m.retrieval_latency_ms == 42.5
        assert m.retrieved_chunk_count == 5
        assert m.distance_metric == "unknown"
        assert m.unique_url_count == 0
        assert m.unique_domain_count == 0
        assert m.average_semantic_score == 0.0
        assert m.average_keyword_score == 0.0
        assert m.average_hybrid_score == 0.0
        assert m.source_diversity_ratio_url == 0.0
        assert m.source_diversity_ratio_domain == 0.0
        assert m.detailed_scores is False

    def test_custom_values(self):
        """All fields can be customised."""
        m = RetrievalMetrics(
            retrieval_latency_ms=10.0,
            retrieved_chunk_count=3,
            distance_metric="cosine",
            unique_url_count=3,
            unique_domain_count=2,
            average_hybrid_score=0.7,
            source_diversity_ratio_url=1.0,
            source_diversity_ratio_domain=0.667,
        )
        assert m.distance_metric == "cosine"
        assert m.unique_url_count == 3
        assert m.unique_domain_count == 2
        assert m.source_diversity_ratio_url == 1.0


class TestFailedQuery:
    """Tests for the FailedQuery dataclass."""

    def test_default_query_id(self):
        """query_id defaults to empty string."""
        fq = FailedQuery(query="test", error="fail")
        assert fq.query == "test"
        assert fq.error == "fail"
        assert fq.query_id == ""


# ==========================================
# METRIC COMPUTATION TESTS
# ==========================================


class TestComputeMetricsFromResults:
    """Tests for compute_metrics_from_results()."""

    def test_computes_correctly_cosine(self):
        """All metrics computed from result data with cosine distance."""
        ids = ["r:s:1", "r:s:2", "r:s:3"]
        metadatas = [
            {"url": "https://cnn.com/a"},
            {"url": "https://cnn.com/b"},
            {"url": "https://reuters.com/c"},
        ]
        distances = [0.1, 0.3, 0.5]

        metrics = compute_metrics_from_results(
            latency_ms=15.0,
            ids=ids,
            metadatas=metadatas,
            distances=distances,
            distance_metric="cosine",
        )

        assert metrics.retrieval_latency_ms == 15.0
        assert metrics.retrieved_chunk_count == 3
        assert metrics.distance_metric == "cosine"
        assert metrics.unique_url_count == 3
        assert metrics.unique_domain_count == 2
        assert metrics.average_hybrid_score == pytest.approx(
            ((1.0 - 0.1) + (1.0 - 0.3) + (1.0 - 0.5)) / 3, 0.001
        )
        assert metrics.source_diversity_ratio_url == pytest.approx(
            3 / 3, 0.001
        )
        assert metrics.source_diversity_ratio_domain == pytest.approx(
            2 / 3, 0.001
        )

    def test_l2_metric_hybrid_zero(self):
        """Non-cosine metric yields zero hybrid score."""
        ids = ["r:s:1"]
        metadatas = [{"url": "https://a.com"}]
        distances = [0.1]

        metrics = compute_metrics_from_results(
            latency_ms=5.0,
            ids=ids,
            metadatas=metadatas,
            distances=distances,
            distance_metric="l2",
        )
        assert metrics.average_hybrid_score == 0.0
        assert metrics.distance_metric == "l2"

    def test_empty_results(self):
        """Empty lists produce zeroed metrics."""
        metrics = compute_metrics_from_results(
            latency_ms=5.0,
            ids=[],
            metadatas=[],
            distances=[],
        )
        assert metrics.retrieved_chunk_count == 0
        assert metrics.unique_url_count == 0
        assert metrics.unique_domain_count == 0
        assert metrics.average_hybrid_score == 0.0
        assert metrics.source_diversity_ratio_url == 0.0
        assert metrics.source_diversity_ratio_domain == 0.0

    def test_diversity_ratio_single_source(self):
        """Single source yields diversity ratio of 1.0 for both levels."""
        ids = ["r:s:1"]
        metadatas = [{"url": "https://a.com"}]
        distances = [0.2]

        metrics = compute_metrics_from_results(
            latency_ms=5.0,
            ids=ids,
            metadatas=metadatas,
            distances=distances,
            distance_metric="cosine",
        )
        assert metrics.unique_url_count == 1
        assert metrics.unique_domain_count == 1
        assert metrics.source_diversity_ratio_url == 1.0
        assert metrics.source_diversity_ratio_domain == 1.0

    def test_missing_url_in_metadata(self):
        """Chunks without a URL do not contribute to counts."""
        ids = ["r:s:1", "r:s:2"]
        metadatas = [{}, {"url": ""}]
        distances = [0.1, 0.2]

        metrics = compute_metrics_from_results(
            latency_ms=10.0, ids=ids, metadatas=metadatas, distances=distances
        )
        assert metrics.unique_url_count == 0
        assert metrics.unique_domain_count == 0

    def test_detailed_scores_flag(self):
        """detailed_scores is passed through."""
        metrics = compute_metrics_from_results(
            latency_ms=5.0,
            ids=["r:s:1"],
            metadatas=[{"url": "https://a.com"}],
            distances=[0.1],
            distance_metric="cosine",
            detailed_scores=True,
        )
        assert metrics.detailed_scores is True


class TestAggregateMetrics:
    """Tests for aggregate_metrics()."""

    def make_result(
        self,
        latency,
        chunks,
        urls,
        domains,
        scores,
        metric="cosine",
    ):
        """Helper to build an EvaluationResult."""
        top_chunks = [
            ChunkScore(
                chunk_id=f"r:s:{i}",
                raw_distance=1.0 - s,
                distance_metric=metric,
            )
            for i, s in enumerate(scores)
        ]
        metrics = RetrievalMetrics(
            retrieval_latency_ms=latency,
            retrieved_chunk_count=chunks,
            distance_metric=metric,
            unique_url_count=urls,
            unique_domain_count=domains,
            average_hybrid_score=(
                round(sum(scores) / len(scores), 4) if scores else 0.0
            ),
            source_diversity_ratio_url=round(urls / chunks, 4) if chunks else 0.0,
            source_diversity_ratio_domain=(
                round(domains / chunks, 4) if chunks else 0.0
            ),
        )
        return EvaluationResult(
            query="q",
            latency_ms=latency,
            retrieved_chunk_count=chunks,
            unique_url_count=urls,
            unique_domain_count=domains,
            top_chunks=top_chunks,
            metrics=metrics,
        )

    def test_averages_across_results(self):
        """Metrics are correctly averaged across results."""
        r1 = self.make_result(
            latency=10.0, chunks=4, urls=2, domains=1, scores=[0.9, 0.8, 0.7, 0.6]
        )
        r2 = self.make_result(
            latency=20.0, chunks=2, urls=2, domains=2, scores=[0.9, 0.8]
        )

        aggregated = aggregate_metrics([r1, r2])

        assert aggregated.retrieval_latency_ms == 15.0
        assert aggregated.retrieved_chunk_count == 3
        assert aggregated.unique_url_count == 2
        assert aggregated.unique_domain_count == 2  # round(mean([1, 2])) = 2
        assert aggregated.average_hybrid_score == pytest.approx(4.7 / 6, 0.001)

    def test_empty_results(self):
        """Empty input yields zeroed metrics."""
        aggregated = aggregate_metrics([])
        assert aggregated.retrieval_latency_ms == 0.0
        assert aggregated.retrieved_chunk_count == 0

    def test_detailed_scores_flag(self):
        """detailed_scores is True if any result has it."""
        r = self.make_result(
            latency=5.0, chunks=1, urls=1, domains=1, scores=[0.9]
        )
        # Patch detailed_scores
        r = EvaluationResult(
            query="q",
            latency_ms=5.0,
            retrieved_chunk_count=1,
            unique_url_count=1,
            unique_domain_count=1,
            metrics=RetrievalMetrics(
                retrieval_latency_ms=5.0,
                retrieved_chunk_count=1,
                distance_metric="cosine",
                unique_url_count=1,
                unique_domain_count=1,
                detailed_scores=True,
            ),
        )
        aggregated = aggregate_metrics([r])
        assert aggregated.detailed_scores is True

    def test_mixed_metrics(self):
        """L2 results don't contribute to hybrid score."""
        r1 = self.make_result(
            latency=10.0, chunks=2, urls=2, domains=2, scores=[0.9, 0.8], metric="cosine"
        )
        r2 = self.make_result(
            latency=20.0, chunks=2, urls=1, domains=1, scores=[0.0, 0.0], metric="l2"
        )

        aggregated = aggregate_metrics([r1, r2])

        # Only cosine results contribute to hybrid
        assert aggregated.average_hybrid_score == pytest.approx(0.85, 0.01)
        # Distance metric should be cosine (majority)
        assert aggregated.distance_metric == "cosine"


# ==========================================
# RETRIEVAL EVALUATOR TESTS
# ==========================================


class MockRetrievalFn:
    """Simulates retrieve_chunks for deterministic testing."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, query, top_k=5):
        self.call_count += 1
        return {
            "ids": [
                [
                    "rep1:src1:1",
                    "rep1:src1:2",
                    "rep1:src2:3",
                ]
            ],
            "documents": [
                [
                    "Content for chunk one.",
                    "Content for chunk two.",
                    "Content for chunk three.",
                ]
            ],
            "metadatas": [
                [
                    {
                        "chunk_id": 1,
                        "url": "https://cnn.com/article-a",
                        "title": "Article A",
                    },
                    {
                        "chunk_id": 2,
                        "url": "https://cnn.com/article-b",
                        "title": "Article B",
                    },
                    {
                        "chunk_id": 3,
                        "url": "https://reuters.com/article-c",
                        "title": "Article C",
                    },
                ]
            ],
            "distances": [[0.1, 0.3, 0.5]],
        }


class EmptyMockRetrievalFn:
    """Returns no results (simulates empty corpus)."""

    def __call__(self, query, top_k=5):
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }


class FailingMockRetrievalFn:
    """Simulates a retrieval failure."""

    def __call__(self, query, top_k=5):
        raise RuntimeError("Simulated retrieval failure")


class TestRetrievalEvaluator:
    """Tests for RetrievalEvaluator."""

    def test_init_default_retrieval_fn(self):
        """Default constructor uses retrieve_chunks."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )
        from app.services.retrieval_service import retrieve_chunks

        evaluator = RetrievalEvaluator()
        assert evaluator.retrieval_fn is retrieve_chunks
        assert evaluator.default_top_k == 5

    def test_init_detects_cosine_metric(self):
        """Default constructor detects cosine distance for retrieve_chunks."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator()
        assert evaluator.distance_metric == "cosine"

    def test_init_custom_metric_override(self):
        """distance_metric can be overridden."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn(),
            distance_metric="l2",
        )
        assert evaluator.distance_metric == "l2"

    def test_init_custom_retrieval_fn(self):
        """Custom retrieval function is used."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(
            retrieval_fn=fn, default_top_k=3
        )
        assert evaluator.retrieval_fn is fn
        assert evaluator.default_top_k == 3

    def test_evaluate_query_returns_evaluation_result(self):
        """evaluate_query returns an EvaluationResult."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn()
        )
        result = evaluator.evaluate_query("test query")

        assert isinstance(result, EvaluationResult)
        assert result.query == "test query"

    def test_evaluate_query_populates_metrics(self):
        """All metrics are populated from retrieval results."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn(),
            distance_metric="cosine",
        )
        result = evaluator.evaluate_query("test query", top_k=3)

        assert result.retrieved_chunk_count == 3
        assert result.unique_url_count == 3
        assert result.unique_domain_count == 2  # cnn.com + reuters.com
        assert result.latency_ms > 0
        assert len(result.top_chunks) == 3
        assert result.metrics is not None
        assert result.metrics.retrieved_chunk_count == 3
        assert result.metrics.unique_url_count == 3
        assert result.metrics.unique_domain_count == 2
        assert result.metrics.distance_metric == "cosine"

    def test_evaluate_query_deterministic(self):
        """Same query produces identical results."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        mock_fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(retrieval_fn=mock_fn)

        r1 = evaluator.evaluate_query("test query", top_k=3)
        r2 = evaluator.evaluate_query("test query", top_k=3)

        assert r1.retrieved_chunk_count == r2.retrieved_chunk_count
        assert r1.unique_url_count == r2.unique_url_count
        assert r1.unique_domain_count == r2.unique_domain_count
        for c1, c2 in zip(r1.top_chunks, r2.top_chunks):
            assert c1.chunk_id == c2.chunk_id
            assert c1.raw_distance == c2.raw_distance

    def test_evaluate_query_empty_raises(self):
        """Empty query raises ValueError."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn()
        )

        with pytest.raises(ValueError, match="non-empty"):
            evaluator.evaluate_query("")

    def test_evaluate_query_whitespace_raises(self):
        """Whitespace-only query raises ValueError."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn()
        )

        with pytest.raises(ValueError, match="non-empty"):
            evaluator.evaluate_query("   ")

    def test_evaluate_query_with_id_and_category(self):
        """query_id and category are passed through to result."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn()
        )
        result = evaluator.evaluate_query(
            query="test",
            top_k=2,
            query_id="hist-001",
            category="history",
        )

        assert result.query_id == "hist-001"
        assert result.category == "history"

    def test_empty_results(self):
        """Empty retrieval results produce zeroed metrics."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=EmptyMockRetrievalFn()
        )
        result = evaluator.evaluate_query("test", top_k=5)

        assert result.retrieved_chunk_count == 0
        assert result.unique_url_count == 0
        assert result.unique_domain_count == 0
        assert len(result.top_chunks) == 0
        assert result.metrics is not None
        assert result.metrics.retrieved_chunk_count == 0

    def test_retrieval_failure_graceful(self):
        """Retrieval failure returns zeroed result (no exception)."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=FailingMockRetrievalFn()
        )
        result = evaluator.evaluate_query("test", top_k=5)

        assert result.retrieved_chunk_count == 0
        assert result.latency_ms == 0.0

    def test_evaluate_queries_multiple(self):
        """evaluate_queries returns results for all valid queries."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        mock_fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(retrieval_fn=mock_fn)

        queries = ["q1", "q2", "q3"]
        results, failures = evaluator.evaluate_queries(
            queries, top_k=2
        )

        assert len(results) == 3
        assert len(failures) == 0
        assert mock_fn.call_count == 3

    def test_evaluate_queries_skips_empty(self):
        """Empty query is skipped without killing the batch."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        mock_fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(retrieval_fn=mock_fn)

        queries = ["valid", "", "also valid"]
        results, failures = evaluator.evaluate_queries(
            queries, top_k=2
        )

        assert len(results) == 2
        assert len(failures) == 1
        assert failures[0].query == ""
        assert "non-empty" in failures[0].error

    def test_evaluate_queries_empty_list(self):
        """Empty query list yields empty results/failures."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn()
        )
        results, failures = evaluator.evaluate_queries([])
        assert results == []
        assert failures == []

    # ---------------------------------------------------------------
    # BENCHMARK TESTS
    # ---------------------------------------------------------------

    def test_evaluate_benchmark_defaults(self):
        """evaluate_benchmark uses DEFAULT_BENCHMARK_QUERIES."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        mock_fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(retrieval_fn=mock_fn)

        report = evaluator.evaluate_benchmark()

        assert isinstance(report, BenchmarkReport)
        assert report.total_queries == 35
        assert report.successful_queries == 35
        assert report.failed_queries == 0
        assert len(report.per_query_results) == 35
        assert report.overall_metrics is not None

    def test_evaluate_benchmark_custom_queries(self):
        """evaluate_benchmark accepts a custom query list."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        mock_fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(retrieval_fn=mock_fn)

        custom = [
            BenchmarkQuery(
                id="custom-001",
                query="custom query",
                category="test",
            ),
        ]
        report = evaluator.evaluate_benchmark(
            queries=custom, top_k=3
        )

        assert report.total_queries == 1
        assert report.successful_queries == 1
        assert len(report.per_query_results) == 1
        assert report.per_query_results[0].query == "custom query"
        assert report.per_query_results[0].query_id == "custom-001"
        assert report.queries_by_category == {"test": 1}

    def test_evaluate_benchmark_queries_by_category(self):
        """queries_by_category is correctly populated."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        mock_fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(retrieval_fn=mock_fn)

        queries = [
            BenchmarkQuery(
                id="h1", query="q1", category="history"
            ),
            BenchmarkQuery(
                id="h2", query="q2", category="history"
            ),
            BenchmarkQuery(
                id="s1", query="q3", category="science"
            ),
        ]
        report = evaluator.evaluate_benchmark(
            queries=queries, top_k=2
        )

        assert report.queries_by_category == {
            "history": 2,
            "science": 1,
        }

    def test_evaluate_benchmark_skips_invalid_queries(self):
        """Benchmark skips invalid queries and tracks failures."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn()
        )

        queries = [
            BenchmarkQuery(
                id="valid-1", query="valid", category="test"
            ),
            BenchmarkQuery(
                id="empty-1",
                query="",
                category="test",
            ),
            BenchmarkQuery(
                id="valid-2", query="also valid", category="test"
            ),
        ]
        report = evaluator.evaluate_benchmark(
            queries=queries, top_k=2
        )

        assert report.total_queries == 3
        assert report.successful_queries == 2
        assert report.failed_queries == 1
        assert len(report.failure_details) == 1
        assert report.failure_details[0].query_id == "empty-1"

    # ---------------------------------------------------------------
    # STABILITY TESTS
    # ---------------------------------------------------------------

    def test_evaluate_stability_deterministic(self):
        """Fully deterministic retrieval yields stability of 1.0."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        mock_fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(retrieval_fn=mock_fn)

        stability = evaluator.evaluate_stability(
            "test", runs=3, top_k=3
        )

        assert stability == 1.0

    def test_evaluate_stability_empty_query_raises(self):
        """Stability with empty query raises ValueError."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn()
        )

        with pytest.raises(ValueError):
            evaluator.evaluate_stability("", runs=3)

    def test_evaluate_stability_single_run(self):
        """Single run returns 1.0 (trivially stable)."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(
            retrieval_fn=MockRetrievalFn()
        )

        stability = evaluator.evaluate_stability(
            "test", runs=1, top_k=3
        )
        assert stability == 1.0

    # ---------------------------------------------------------------
    # TOP-K CLAMPING
    # ---------------------------------------------------------------

    def test_top_k_clamping(self):
        """default_top_k is clamped to valid range."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        evaluator = RetrievalEvaluator(default_top_k=50)
        assert evaluator.default_top_k == 20

        evaluator = RetrievalEvaluator(default_top_k=0)
        assert evaluator.default_top_k == 1

    def test_top_k_override(self):
        """Passing top_k to evaluate_query overrides default."""
        from app.evaluation.retrieval_evaluator import (
            RetrievalEvaluator,
        )

        mock_fn = MockRetrievalFn()
        evaluator = RetrievalEvaluator(
            retrieval_fn=mock_fn, default_top_k=5
        )
        result = evaluator.evaluate_query("test", top_k=3)
        assert result.retrieved_chunk_count == 3


# ==========================================
# IMPORT TESTS
# ==========================================


class TestPackageImports:
    """Tests that package exports work correctly."""

    def test_from_evaluation_import(self):
        """Import from app.evaluation works."""
        from app.evaluation import (
            RetrievalEvaluator,
            DEFAULT_BENCHMARK_QUERIES,
            BenchmarkQuery,
            RetrievalMetrics,
            ChunkScore,
            EvaluationResult,
            BenchmarkReport,
            FailedQuery,
            compute_jaccard_similarity,
            compute_stability,
        )
        assert RetrievalEvaluator is not None
        assert len(DEFAULT_BENCHMARK_QUERIES) == 35
        assert BenchmarkQuery is not None
        assert RetrievalMetrics is not None
        assert ChunkScore is not None
        assert EvaluationResult is not None
        assert BenchmarkReport is not None
        assert FailedQuery is not None
        assert callable(compute_jaccard_similarity)
        assert callable(compute_stability)

    def test_datasets_export(self):
        """Import from datasets directly works."""
        from app.evaluation.datasets import (
            DEFAULT_BENCHMARK_QUERIES,
            BENCHMARK_FAILURE_QUERIES,
            BenchmarkQuery,
        )
        assert len(DEFAULT_BENCHMARK_QUERIES) == 35
        assert len(BENCHMARK_FAILURE_QUERIES) == 5
        assert BenchmarkQuery is not None

    def test_metrics_export(self):
        """Import from metrics directly works."""
        from app.evaluation.metrics import (
            RetrievalMetrics,
            ChunkScore,
            EvaluationResult,
            BenchmarkReport,
            FailedQuery,
            compute_metrics_from_results,
            aggregate_metrics,
            compute_jaccard_similarity,
            compute_stability,
        )
        assert RetrievalMetrics is not None
        assert ChunkScore is not None
        assert EvaluationResult is not None
        assert BenchmarkReport is not None
        assert FailedQuery is not None
        assert callable(compute_metrics_from_results)
        assert callable(aggregate_metrics)
        assert callable(compute_jaccard_similarity)
        assert callable(compute_stability)
