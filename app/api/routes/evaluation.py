# ==========================================
# EVALUATION API ENDPOINTS
# ==========================================
#
# Exposes the evaluation frameworks (retrieval
# and evidence) via the public API without
# modifying any production workflows.
#
# Endpoints:
#   POST /api/v1/evaluate/retrieval
#   POST /api/v1/evaluate/evidence
#
# ==========================================

import time
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.schemas.evaluation_schema import (
    EvidenceEvaluationRequest,
    RetrievalEvaluationRequest,
)
from app.evaluation.evidence_evaluator import (
    EvidenceEvaluationResult,
    evaluate_evidence,
)
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.datasets import DEFAULT_BENCHMARK_QUERIES
from app.utils.logger import logger
from app.utils.response_builder import error_response, success_response

from dataclasses import asdict
from typing import Optional

from app.repositories.evaluation_repository import (
    get_recent_evaluations,
    save_evaluation,
)

router = APIRouter()


# ==========================================
# HELPERS
# ==========================================


def _get_request_id(request: Request) -> str:
    """Extract the request_id from the request state."""
    return str(getattr(request.state, "request_id", "unknown"))


def _dataclass_to_dict(obj) -> dict:
    """Convert a (possibly nested) dataclass to a plain dict."""
    return asdict(obj)


# ==========================================
# POST /api/v1/evaluate/retrieval
# ==========================================


@router.post(
    "/api/v1/evaluate/retrieval",
    summary="Evaluate Retrieval Quality",
    description=(
        "Run a batch of queries through the retrieval system and return "
        "quality metrics including latency, source diversity, distance "
        "metrics, and stability. Invalid queries (empty/whitespace) are "
        "gracefully skipped and reported in failure_details."
    ),
    responses={
        200: {
            "description": "Successful retrieval evaluation",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Retrieval evaluation completed",
                        "timestamp": "2026-06-12T12:00:00.000Z",
                        "request_id": "req-abc-123",
                        "data": {
                            "total_queries": 2,
                            "successful_queries": 2,
                            "failed_queries": 0,
                            "overall_metrics": {
                                "retrieval_latency_ms": 0.05,
                                "retrieved_chunk_count": 5,
                                "distance_metric": "cosine",
                                "unique_url_count": 4,
                                "unique_domain_count": 3,
                                "average_hybrid_score": 0.72,
                                "source_diversity_ratio_url": 0.8,
                                "source_diversity_ratio_domain": 0.6,
                            },
                            "failure_details": [],
                            "per_query_results": [],
                        },
                    }
                }
            },
        },
        422: {"description": "Validation error (empty queries list, invalid top_k)"},
        500: {"description": "Internal evaluation error"},
    },
)
async def evaluate_retrieval(
    payload: RetrievalEvaluationRequest,
    request: Request,
):
    """Evaluate retrieval quality for one or more search queries.

    Returns per-query metrics and an aggregated benchmark report
    including latency, source diversity, and distance metrics.
    """
    request_id = _get_request_id(request)
    logger.info(
        "[EVAL API] POST /evaluate/retrieval | "
        "request_id=%s | queries=%d | top_k=%s",
        request_id,
        len(payload.queries),
        payload.top_k,
    )

    start_time = time.time()

    try:
        evaluator = RetrievalEvaluator(default_top_k=payload.top_k or 5)
        results, failures = evaluator.evaluate_queries(payload.queries, top_k=payload.top_k)

        # Build a lightweight report from the results
        total = len(payload.queries)
        successful = len(results)
        failed = len(failures)

        overall_metrics = None
        if results:
            from app.evaluation.metrics import aggregate_metrics
            overall_metrics = _dataclass_to_dict(aggregate_metrics(results))

        per_query = [_dataclass_to_dict(r) for r in results]
        failure_details = [_dataclass_to_dict(f) for f in failures]

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        data = {
            "total_queries": total,
            "successful_queries": successful,
            "failed_queries": failed,
            "overall_metrics": overall_metrics,
            "failure_details": failure_details,
            "per_query_results": per_query,
            "evaluation_time_ms": elapsed_ms,
        }

        logger.info(
            "[EVAL API] Retrieval evaluation complete | "
            "request_id=%s | %d/%d passed | %.2fms",
            request_id,
            successful,
            total,
            elapsed_ms,
        )

        # ── Persist ──
        # Persistence failures are logged as warnings and never
        # affect the response or propagate to the caller.
        try:
            await save_evaluation(
                evaluation_type="retrieval",
                request_id=request_id,
                results=data,
            )
        except Exception as persist_e:
            logger.warning(
                "[EVAL API] Persistence skipped (retrieval): %s",
                persist_e,
            )

        return success_response(
            data=data,
            message="Retrieval evaluation completed",
            timestamp=str(datetime.utcnow()),
            request_id=request_id,
        )

    except Exception as e:
        logger.error(
            "[EVAL API] Retrieval evaluation failed: %s | request_id=%s",
            e,
            request_id,
        )
        return JSONResponse(
            status_code=500,
            content=error_response(
                message="Retrieval evaluation failed",
                error_code="EVALUATION_ERROR",
                details=str(e),
            ),
        )


# ==========================================
# POST /api/v1/evaluate/evidence
# ==========================================


@router.post(
    "/api/v1/evaluate/evidence",
    summary="Evaluate Evidence Quality",
    description=(
        "Evaluate the quality and grounding of an evidence output. "
        "Returns metrics including support ratio, confidence distribution, "
        "source diversity, and coverage score. Accepts multiple input "
        "schemas and handles malformed inputs gracefully."
    ),
    responses={
        200: {
            "description": "Successful evidence evaluation",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Evidence evaluation completed",
                        "timestamp": "2026-06-12T12:00:00.000Z",
                        "request_id": "req-abc-123",
                        "data": {
                            "total_facts": 3,
                            "supported_fact_count": 2,
                            "unsupported_fact_count": 1,
                            "average_fact_confidence": 0.71,
                            "citation_count": 4,
                            "unique_source_count": 3,
                            "unique_domain_count": 2,
                            "support_ratio": 0.67,
                            "evidence_per_fact": 1.33,
                            "coverage_score": 0.5,
                            "confidence_distribution": {
                                "0.00-0.20": 0,
                                "0.21-0.40": 0,
                                "0.41-0.60": 1,
                                "0.61-0.80": 1,
                                "0.81-1.00": 1,
                            },
                        },
                    }
                }
            },
        },
        422: {"description": "Validation error (missing evidence payload)"},
        500: {"description": "Internal evaluation error"},
    },
)
async def evaluate_evidence_endpoint(
    payload: EvidenceEvaluationRequest,
    request: Request,
):
    """Evaluate the quality of an evidence output.

    Processes facts and their supporting evidence to compute metrics
    such as support ratio, confidence distribution, source diversity,
    and coverage score.
    """
    request_id = _get_request_id(request)
    logger.info(
        "[EVAL API] POST /evaluate/evidence | request_id=%s",
        request_id,
    )

    start_time = time.time()

    try:
        result: EvidenceEvaluationResult = evaluate_evidence(payload.evidence)

        data = _dataclass_to_dict(result)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        data["evaluation_time_ms"] = elapsed_ms

        logger.info(
            "[EVAL API] Evidence evaluation complete | "
            "request_id=%s | %d facts, %.2f%% support ratio | %.2fms",
            request_id,
            result.total_facts,
            result.support_ratio * 100,
            elapsed_ms,
        )

        # ── Persist ──
        try:
            await save_evaluation(
                evaluation_type="evidence",
                request_id=request_id,
                results=data,
            )
        except Exception as persist_e:
            logger.warning(
                "[EVAL API] Persistence skipped (evidence): %s",
                persist_e,
            )

        return success_response(
            data=data,
            message="Evidence evaluation completed",
            timestamp=str(datetime.utcnow()),
            request_id=request_id,
        )

    except Exception as e:
        logger.error(
            "[EVAL API] Evidence evaluation failed: %s | request_id=%s",
            e,
            request_id,
        )
        return JSONResponse(
            status_code=500,
            content=error_response(
                message="Evidence evaluation failed",
                error_code="EVALUATION_ERROR",
                details=str(e),
            ),
        )


# ==========================================
# GET /api/v1/evaluate/summary
# ==========================================


@router.get(
    "/api/v1/evaluate/summary",
    summary="Evaluation Summary Snapshot",
    description=(
        "Return a high-level snapshot of system evaluation status. "
        "Includes retrieval benchmark metrics (query count, average "
        "latency, stability, domain diversity), evidence quality "
        "metrics (if available), and system metadata. Metrics that "
        "cannot be computed are returned as null."
    ),
    responses={
        200: {
            "description": "Successful evaluation summary",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Evaluation summary",
                        "request_id": "req-abc-123",
                        "data": {
                            "retrieval": {
                                "benchmark_queries": 35,
                                "average_latency_ms": 0.05,
                                "average_stability": 0.95,
                                "average_domain_diversity": 0.6,
                                "status": "available",
                            },
                            "evidence": {
                                "average_support_ratio": None,
                                "average_coverage_score": None,
                                "average_confidence": None,
                                "status": "unavailable",
                            },
                            "system": {
                                "total_tests": 232,
                            },
                        },
                    }
                }
            },
        },
        500: {"description": "Internal error"},
    },
)
async def evaluate_summary(
    request: Request,
):
    """Return an evaluation summary snapshot.

    Computes retrieval benchmark metrics from the default query set
    and reports evidence quality metrics as null when no evidence
    input has been provided. The endpoint is designed for dashboard
    consumption and never raises exceptions.
    """
    request_id = _get_request_id(request)
    logger.info(
        "[EVAL API] GET /evaluate/summary | request_id=%s",
        request_id,
    )

    start_time = time.time()

    # ----------------------------------------------------------
    # RETRIEVAL — run the benchmark + stability on a sample query
    # ----------------------------------------------------------

    benchmark_queries_count = len(DEFAULT_BENCHMARK_QUERIES)
    avg_latency: Optional[float] = None
    avg_stability: Optional[float] = None
    avg_domain_diversity: Optional[float] = None
    retrieval_status = "unavailable"

    # Try benchmark first
    try:
        evaluator = RetrievalEvaluator(default_top_k=5)
        benchmark = evaluator.evaluate_benchmark()

        if benchmark.overall_metrics is not None:
            avg_latency = benchmark.overall_metrics.retrieval_latency_ms
            avg_domain_diversity = (
                benchmark.overall_metrics.source_diversity_ratio_domain
            )

        retrieval_status = "available"

        # Stability: try independently so a stability failure
        # doesn't null out the benchmark metrics
        if benchmark.per_query_results:
            try:
                sample_query = benchmark.per_query_results[0].query
                avg_stability = evaluator.evaluate_stability(
                    query=sample_query, runs=3, top_k=5
                )
            except Exception as stab_e:
                logger.warning(
                    "[EVAL API] Stability unavailable: %s", stab_e,
                )
                avg_stability = None

    except Exception as e:
        logger.warning(
            "[EVAL API] Retrieval summary unavailable: %s", e,
        )
        retrieval_status = "unavailable"

    retrieval_section = {
        "benchmark_queries": benchmark_queries_count,
        "average_latency_ms": avg_latency,
        "average_stability": avg_stability,
        "average_domain_diversity": avg_domain_diversity,
        "status": retrieval_status,
    }

    # ----------------------------------------------------------
    # EVIDENCE — no default input available, mark as unavailable
    # ----------------------------------------------------------

    evidence_section = {
        "average_support_ratio": None,
        "average_coverage_score": None,
        "average_confidence": None,
        "status": "unavailable",
    }

    # ----------------------------------------------------------
    # SYSTEM
    # ----------------------------------------------------------

    system_section = {
        "total_tests": 232,
    }

    # ----------------------------------------------------------
    # ASSEMBLE
    # ----------------------------------------------------------

    elapsed_ms = round((time.time() - start_time) * 1000, 2)

    data = {
        "retrieval": retrieval_section,
        "evidence": evidence_section,
        "system": system_section,
        "evaluation_time_ms": elapsed_ms,
    }

    logger.info(
        "[EVAL API] Summary complete | request_id=%s | "
        "retrieval=%s | %.2fms",
        request_id,
        retrieval_section.get("status", "unknown"),
        elapsed_ms,
    )

    return success_response(
        data=data,
        message="Evaluation summary",
        timestamp=str(datetime.utcnow()),
        request_id=request_id,
    )


# ==========================================
# GET /api/v1/evaluate/history
# ==========================================


@router.get(
    "/api/v1/evaluate/history",
    summary="Evaluation History",
    description=(
        "Retrieve past evaluation runs in reverse chronological order. "
        "Supports pagination via ``limit`` and ``skip`` query parameters. "
        "Failures to connect to MongoDB return an empty list gracefully."
    ),
    responses={
        200: {
            "description": "Successful evaluation history retrieval",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "message": "Evaluation history retrieved",
                        "request_id": "req-abc-123",
                        "data": {
                            "evaluations": [
                                {
                                    "_id": "abc123...",
                                    "evaluation_type": "retrieval",
                                    "request_id": "req-xyz-456",
                                    "created_at": "2026-06-12T12:00:00",
                                    "results": {
                                        "total_queries": 2,
                                        "successful_queries": 2,
                                    },
                                }
                            ],
                            "count": 1,
                            "limit": 20,
                            "skip": 0,
                        },
                    }
                }
            },
        },
        500: {"description": "Internal error"},
    },
)
async def evaluate_history(
    request: Request,
    limit: int = 20,
    skip: int = 0,
):
    """Retrieve past evaluation runs.

    Args:
        limit: Maximum number of results to return (default 20, max 100).
        skip:  Number of results to skip (for pagination).

    Returns:
        A list of evaluation documents with pagination metadata.
        Returns an empty list on Mongo failure (never raises).
    """
    request_id = _get_request_id(request)
    clamped_limit = max(1, min(100, limit))
    clamped_skip = max(0, skip)

    logger.info(
        "[EVAL API] GET /evaluate/history | request_id=%s | "
        "limit=%d, skip=%d",
        request_id,
        clamped_limit,
        clamped_skip,
    )

    try:
        evaluations = await get_recent_evaluations(
            limit=clamped_limit,
            skip=clamped_skip,
        )

        data = {
            "evaluations": evaluations,
            "count": len(evaluations),
            "limit": clamped_limit,
            "skip": clamped_skip,
        }

        logger.info(
            "[EVAL API] History retrieved | request_id=%s | %d items",
            request_id,
            len(evaluations),
        )

        return success_response(
            data=data,
            message="Evaluation history retrieved",
            timestamp=str(datetime.utcnow()),
            request_id=request_id,
        )

    except Exception as e:
        logger.error(
            "[EVAL API] History retrieval failed: %s | request_id=%s",
            e,
            request_id,
        )
        return JSONResponse(
            status_code=500,
            content=error_response(
                message="History retrieval failed",
                error_code="HISTORY_ERROR",
                details=str(e),
            ),
        )
