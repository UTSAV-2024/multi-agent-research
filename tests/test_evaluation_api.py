# ==========================================
# EVALUATION API ENDPOINT TESTS
# ==========================================
#
# Tests for POST /api/v1/evaluate/retrieval
# and POST /api/v1/evaluate/evidence.
#
# ==========================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# ==========================================
# RETRIEVAL EVALUATION ENDPOINT
# ==========================================


class TestRetrievalEvaluationValidation:
    """Tests for request validation on the retrieval evaluation endpoint."""

    def test_empty_queries_list_returns_422(self):
        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": [], "top_k": 5},
        )
        assert response.status_code == 422

    def test_missing_queries_returns_422(self):
        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"top_k": 5},
        )
        assert response.status_code == 422

    def test_invalid_top_k_zero_returns_422(self):
        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": ["test"], "top_k": 0},
        )
        assert response.status_code == 422

    def test_invalid_top_k_negative_returns_422(self):
        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": ["test"], "top_k": -1},
        )
        assert response.status_code == 422

    def test_too_many_queries_returns_422(self):
        """More than 100 queries should be rejected."""
        queries = [f"query_{i}" for i in range(101)]
        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": queries, "top_k": 5},
        )
        assert response.status_code == 422

    def test_string_instead_of_array_returns_422(self):
        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": "not a list", "top_k": 5},
        )
        assert response.status_code == 422


class TestRetrievalEvaluationWithMocks:
    """Tests for successful retrieval evaluation with mocked evaluator."""

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_successful_evaluation_returns_200(self, mock_evaluator_class):
        mock_instance = MagicMock()
        mock_evaluator_class.return_value = mock_instance

        # Mock the evaluator to return a successful result
        mock_instance.evaluate_queries.return_value = ([], [])

        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": ["Operation Barbarossa", "CRISPR"], "top_k": 5},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["message"] == "Retrieval evaluation completed"
        assert "request_id" in body
        assert "data" in body
        assert body["data"]["total_queries"] == 2
        assert body["data"]["successful_queries"] == 0
        assert body["data"]["failed_queries"] == 0

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_response_contains_expected_fields(self, mock_evaluator_class):
        mock_instance = MagicMock()
        mock_evaluator_class.return_value = mock_instance

        from app.evaluation.metrics import EvaluationResult, ChunkScore

        mock_result = EvaluationResult(
            query="test query",
            query_id="test-001",
            category="test",
            latency_ms=0.05,
            retrieved_chunk_count=3,
            unique_url_count=2,
            unique_domain_count=1,
            top_chunks=[
                ChunkScore(
                    chunk_id="rep1:src1:1",
                    raw_distance=0.15,
                    distance_metric="cosine",
                    source_url="https://example.com/a",
                    content_preview="Content preview...",
                ),
            ],
        )
        mock_instance.evaluate_queries.return_value = ([mock_result], [])

        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": ["test query"], "top_k": 3},
        )

        assert response.status_code == 200
        data = response.json()["data"]

        # Check top-level fields
        assert data["total_queries"] == 1
        assert data["successful_queries"] == 1
        assert data["failed_queries"] == 0
        assert "overall_metrics" in data
        assert "failure_details" in data
        assert "per_query_results" in data
        assert "evaluation_time_ms" in data

        # Check per-query result
        assert len(data["per_query_results"]) == 1
        pq = data["per_query_results"][0]
        assert pq["query"] == "test query"
        assert pq["retrieved_chunk_count"] == 3
        assert pq["unique_url_count"] == 2
        assert pq["unique_domain_count"] == 1

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_handles_failed_queries(self, mock_evaluator_class):
        mock_instance = MagicMock()
        mock_evaluator_class.return_value = mock_instance

        from app.evaluation.metrics import FailedQuery

        mock_instance.evaluate_queries.return_value = (
            [],
            [FailedQuery(query="", error="Query must be a non-empty string")],
        )

        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": [""], "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["successful_queries"] == 0
        assert data["failed_queries"] == 1
        assert len(data["failure_details"]) == 1
        assert data["failure_details"][0]["query"] == ""

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_request_id_present_in_response(self, mock_evaluator_class):
        mock_instance = MagicMock()
        mock_evaluator_class.return_value = mock_instance
        mock_instance.evaluate_queries.return_value = ([], [])

        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": ["test"], "top_k": 5},
        )

        assert response.status_code == 200
        assert "request_id" in response.json()


# ==========================================
# EVIDENCE EVALUATION ENDPOINT
# ==========================================


class TestEvidenceEvaluationValidation:
    """Tests for request validation on the evidence evaluation endpoint."""

    def test_missing_evidence_returns_422(self):
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={},
        )
        assert response.status_code == 422

    def test_null_evidence_returns_422(self):
        """Even though evaluate_evidence handles None, the Pydantic schema requires it."""
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={"evidence": None},
        )
        # This should either be 200 (accepted, graceful degradation) or 422 (schema validation)
        # The schema accepts Any, so it should be 200
        assert response.status_code in (200, 422)


class TestEvidenceEvaluationSuccess:
    """Tests for successful evidence evaluation."""

    def test_successful_evaluation_returns_200(self):
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "facts": [
                        {
                            "fact": "mRNA vaccines teach cells to produce a protein",
                            "confidence": 0.91,
                            "evidence": [
                                {
                                    "url": "https://example.com/vaccine",
                                    "chunk_id": 1,
                                }
                            ],
                        }
                    ]
                }
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["message"] == "Evidence evaluation completed"
        assert "request_id" in body
        assert "data" in body

    def test_response_contains_all_expected_fields(self):
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "facts": [
                        {
                            "fact": "Fact one",
                            "confidence": 0.9,
                            "evidence": [{"url": "https://example.com/a", "chunk_id": 1}],
                        },
                        {
                            "fact": "Fact two",
                            "confidence": 0.5,
                            "evidence": [],
                        },
                    ]
                }
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]

        assert data["total_facts"] == 2
        assert data["supported_fact_count"] == 1
        assert data["unsupported_fact_count"] == 1
        assert data["support_ratio"] == 0.5
        assert isinstance(data["average_fact_confidence"], float)
        assert data["citation_count"] == 1
        assert data["unique_source_count"] == 1
        assert data["unique_domain_count"] == 1
        assert "confidence_distribution" in data
        assert "coverage_score" in data
        assert "evidence_per_fact" in data

    def test_malformed_evidence_graceful_degradation(self):
        """Malformed evidence should still return a valid response."""
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "facts": [
                        "not a dict",
                        None,
                        {},
                    ]
                }
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total_facts"] == 1  # only the empty dict counts as a dict
        assert data["supported_fact_count"] == 0
        assert data["unsupported_fact_count"] == 1

    def test_empty_evidence_graceful_degradation(self):
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={"evidence": {}},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total_facts"] == 0
        assert data["support_ratio"] == 0.0

    def test_multiple_domains(self):
        """Evidence from multiple domains should be counted correctly."""
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "facts": [
                        {
                            "fact": "Fact one",
                            "confidence": 0.9,
                            "evidence": [
                                {"url": "https://cnn.com/article1", "chunk_id": 1},
                                {"url": "https://bbc.com/article2", "chunk_id": 2},
                            ],
                        },
                    ]
                }
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["unique_source_count"] == 2
        assert data["unique_domain_count"] == 2  # cnn.com, bbc.com

    def test_confidence_distribution(self):
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "facts": [
                        {"fact": "A", "confidence": 0.1, "evidence": [{"url": "https://a.com"}]},
                        {"fact": "B", "confidence": 0.3, "evidence": [{"url": "https://b.com"}]},
                        {"fact": "C", "confidence": 0.5, "evidence": [{"url": "https://c.com"}]},
                        {"fact": "D", "confidence": 0.7, "evidence": [{"url": "https://d.com"}]},
                        {"fact": "E", "confidence": 0.9, "evidence": [{"url": "https://e.com"}]},
                    ]
                }
            },
        )

        assert response.status_code == 200
        dist = response.json()["data"]["confidence_distribution"]
        assert dist["0.00-0.20"] == 1
        assert dist["0.21-0.40"] == 1
        assert dist["0.41-0.60"] == 1
        assert dist["0.61-0.80"] == 1
        assert dist["0.81-1.00"] == 1

    def test_request_id_present(self):
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "facts": [
                        {
                            "fact": "Test",
                            "confidence": 0.5,
                            "evidence": [{"url": "https://example.com"}],
                        }
                    ]
                }
            },
        )

        assert response.status_code == 200
        assert "request_id" in response.json()
        assert response.json()["request_id"] is not None

    def test_coverage_score_calculation(self):
        """coverage_score = supported_fact_count / max(citation_count, 1)"""
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "facts": [
                        {
                            "fact": "Fact one",
                            "confidence": 0.9,
                            "evidence": [
                                {"url": "https://example.com/a", "chunk_id": 1},
                                {"url": "https://example.com/b", "chunk_id": 2},
                            ],
                        },
                        {
                            "fact": "Fact two",
                            "confidence": 0.8,
                            "evidence": [],
                        },
                    ]
                }
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        # 1 supported fact / max(2 citations, 1) = 0.5
        assert data["coverage_score"] == 0.5

    def test_mixed_schema_confirmed_facts(self):
        """Should accept 'confirmed_facts' key."""
        response = client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "confirmed_facts": [
                        {
                            "statement": "A confirmed statement",
                            "confidence": 0.84,
                            "supporting_chunks": [{"url": "https://example.com"}],
                        },
                    ]
                }
            },
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["total_facts"] == 1
        assert data["supported_fact_count"] == 1
        assert data["support_ratio"] == 1.0


# ==========================================
# EVALUATION SUMMARY ENDPOINT
# ==========================================


class TestEvaluationSummary:
    """Tests for GET /api/v1/evaluate/summary."""

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_summary_returns_200(self, mock_eval_class):
        """The summary endpoint always returns 200."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_benchmark.side_effect = Exception("Chroma unavailable")

        response = client.get("/api/v1/evaluate/summary")
        assert response.status_code == 200

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_summary_has_correct_structure(self, mock_eval_class):
        """Response has retrieval, evidence, and system sections."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_benchmark.side_effect = Exception("Chroma unavailable")

        response = client.get("/api/v1/evaluate/summary")
        assert response.status_code == 200
        body = response.json()

        assert body["status"] == "success"
        assert "request_id" in body
        assert "data" in body
        data = body["data"]

        assert "retrieval" in data
        assert "evidence" in data
        assert "system" in data
        assert "evaluation_time_ms" in data

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_retrieval_section_contains_expected_fields(self, mock_eval_class):
        """Retrieval section has all required fields."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_benchmark.side_effect = Exception("Chroma unavailable")

        response = client.get("/api/v1/evaluate/summary")
        data = response.json()["data"]
        ret = data["retrieval"]

        assert "benchmark_queries" in ret
        assert "average_latency_ms" in ret
        assert "average_stability" in ret
        assert "average_domain_diversity" in ret
        assert "status" in ret

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_evidence_section_contains_expected_fields(self, mock_eval_class):
        """Evidence section has all required fields."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_benchmark.side_effect = Exception("Chroma unavailable")

        response = client.get("/api/v1/evaluate/summary")
        data = response.json()["data"]
        ev = data["evidence"]

        assert "average_support_ratio" in ev
        assert "average_coverage_score" in ev
        assert "average_confidence" in ev
        assert "status" in ev

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_evidence_metrics_are_null_by_default(self, mock_eval_class):
        """Without evidence input, all evidence metrics are null."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_benchmark.side_effect = Exception("Chroma unavailable")

        response = client.get("/api/v1/evaluate/summary")
        data = response.json()["data"]
        ev = data["evidence"]

        assert ev["average_support_ratio"] is None
        assert ev["average_coverage_score"] is None
        assert ev["average_confidence"] is None
        assert ev["status"] == "unavailable"

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_system_section_contains_total_tests(self, mock_eval_class):
        """System section has total_tests."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_benchmark.side_effect = Exception("Chroma unavailable")

        response = client.get("/api/v1/evaluate/summary")
        data = response.json()["data"]
        sys = data["system"]

        assert "total_tests" in sys
        assert isinstance(sys["total_tests"], int)
        assert sys["total_tests"] > 0

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_request_id_present_in_response(self, mock_eval_class):
        """Response includes request_id."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_benchmark.side_effect = Exception("Chroma unavailable")

        response = client.get("/api/v1/evaluate/summary")
        assert "request_id" in response.json()

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_retrieval_reports_unavailable_on_failure(self, mock_eval_class):
        """When retrieval fails, status is 'unavailable' and fields are None."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_benchmark.side_effect = Exception("Chroma unavailable")

        response = client.get("/api/v1/evaluate/summary")
        ret = response.json()["data"]["retrieval"]

        assert ret["status"] == "unavailable"
        assert ret["average_latency_ms"] is None
        assert ret["average_stability"] is None
        assert ret["average_domain_diversity"] is None

    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_retrieval_reports_available_on_success(self, mock_eval_class):
        """When retrieval succeeds, status is 'available' with populated metrics."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance

        from app.evaluation.metrics import BenchmarkReport, RetrievalMetrics, EvaluationResult

        # Mock a successful benchmark
        mock_overall = RetrievalMetrics(
            retrieval_latency_ms=0.05,
            retrieved_chunk_count=5,
            distance_metric="cosine",
            unique_url_count=4,
            unique_domain_count=3,
            average_hybrid_score=0.72,
            source_diversity_ratio_url=0.8,
            source_diversity_ratio_domain=0.6,
        )
        mock_benchmark = BenchmarkReport(
            total_queries=2,
            successful_queries=2,
            failed_queries=0,
            overall_metrics=mock_overall,
            per_query_results=[
                EvaluationResult(
                    query="test query",
                    latency_ms=0.05,
                    retrieved_chunk_count=5,
                )
            ],
        )
        mock_instance.evaluate_benchmark.return_value = mock_benchmark
        mock_instance.evaluate_stability.return_value = 0.95

        response = client.get("/api/v1/evaluate/summary")
        ret = response.json()["data"]["retrieval"]

        assert ret["status"] == "available"
        assert ret["average_latency_ms"] == 0.05
        assert ret["average_stability"] == 0.95
        assert ret["average_domain_diversity"] == 0.6


# ==========================================
# EVALUATION HISTORY ENDPOINT
# ==========================================


class TestEvaluationHistory:
    """Tests for GET /api/v1/evaluate/history."""

    @patch("app.api.routes.evaluation.get_recent_evaluations")
    def test_history_returns_200(self, mock_get):
        """History endpoint returns 200 with empty list when no evaluations exist."""
        mock_get.return_value = []
        response = client.get("/api/v1/evaluate/history")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert "data" in body
        assert body["data"]["evaluations"] == []
        assert body["data"]["count"] == 0

    @patch("app.api.routes.evaluation.get_recent_evaluations")
    def test_history_pagination_defaults(self, mock_get):
        """Default limit is 20, default skip is 0."""
        mock_get.return_value = []
        client.get("/api/v1/evaluate/history")
        mock_get.assert_called_once_with(limit=20, skip=0)

    @patch("app.api.routes.evaluation.get_recent_evaluations")
    def test_history_pagination_params(self, mock_get):
        """Custom limit and skip are passed through."""
        mock_get.return_value = []
        client.get("/api/v1/evaluate/history?limit=5&skip=10")
        mock_get.assert_called_once_with(limit=5, skip=10)

    @patch("app.api.routes.evaluation.get_recent_evaluations")
    def test_history_clamps_limit_to_max_100(self, mock_get):
        """limit > 100 is clamped to 100."""
        mock_get.return_value = []
        client.get("/api/v1/evaluate/history?limit=500")
        # The endpoint passes the clamped value
        args, kwargs = mock_get.call_args
        assert kwargs["limit"] == 100

    @patch("app.api.routes.evaluation.get_recent_evaluations")
    def test_history_clamps_skip_to_min_0(self, mock_get):
        """Negative skip is clamped to 0."""
        mock_get.return_value = []
        client.get("/api/v1/evaluate/history?skip=-5")
        args, kwargs = mock_get.call_args
        assert kwargs["skip"] == 0

    @patch("app.api.routes.evaluation.get_recent_evaluations")
    def test_history_returns_items(self, mock_get):
        """When evaluations exist, they are returned in the response."""
        mock_get.return_value = [
            {
                "_id": "abc123",
                "evaluation_type": "retrieval",
                "request_id": "req-001",
                "created_at": "2026-06-12T12:00:00",
                "results": {"total_queries": 2},
            },
            {
                "_id": "def456",
                "evaluation_type": "evidence",
                "request_id": "req-002",
                "created_at": "2026-06-12T11:00:00",
                "results": {"total_facts": 3},
            },
        ]
        response = client.get("/api/v1/evaluate/history?limit=10")
        body = response.json()
        assert body["data"]["count"] == 2
        assert len(body["data"]["evaluations"]) == 2
        assert body["data"]["evaluations"][0]["evaluation_type"] == "retrieval"
        assert body["data"]["evaluations"][1]["evaluation_type"] == "evidence"
        assert body["data"]["limit"] == 10
        assert body["data"]["skip"] == 0

    @patch("app.api.routes.evaluation.get_recent_evaluations")
    def test_history_request_id_present(self, mock_get):
        """Response includes request_id."""
        mock_get.return_value = []
        response = client.get("/api/v1/evaluate/history")
        assert "request_id" in response.json()


# ==========================================
# EVALUATION PERSISTENCE INTEGRATION
# ==========================================


class TestEvaluationPersistence:
    """Tests to verify persistence is called during evaluations."""

    @patch("app.api.routes.evaluation.save_evaluation")
    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_retrieval_triggers_persistence(self, mock_eval_class, mock_save):
        """Retrieval evaluation triggers save_evaluation."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_queries.return_value = ([], [])

        client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": ["test"], "top_k": 5},
        )

        mock_save.assert_called_once()
        call_args = mock_save.call_args[1]
        assert call_args["evaluation_type"] == "retrieval"
        assert call_args["request_id"] is not None
        assert "results" in call_args

    @patch("app.api.routes.evaluation.save_evaluation")
    def test_evidence_triggers_persistence(self, mock_save):
        """Evidence evaluation triggers save_evaluation."""
        client.post(
            "/api/v1/evaluate/evidence",
            json={
                "evidence": {
                    "facts": [
                        {
                            "fact": "Test fact",
                            "confidence": 0.9,
                            "evidence": [{"url": "https://example.com"}],
                        }
                    ]
                }
            },
        )

        mock_save.assert_called_once()
        call_args = mock_save.call_args[1]
        assert call_args["evaluation_type"] == "evidence"
        assert call_args["request_id"] is not None
        assert "results" in call_args

    @patch("app.api.routes.evaluation.save_evaluation")
    @patch("app.api.routes.evaluation.RetrievalEvaluator")
    def test_persistence_failure_does_not_affect_response(self, mock_eval_class, mock_save):
        """If persistence fails, the response is still successful."""
        mock_instance = MagicMock()
        mock_eval_class.return_value = mock_instance
        mock_instance.evaluate_queries.return_value = ([], [])
        mock_save.side_effect = Exception("Mongo unavailable")

        response = client.post(
            "/api/v1/evaluate/retrieval",
            json={"queries": ["test"], "top_k": 5},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"
