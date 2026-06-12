# ==========================================
# EVALUATION FRAMEWORK
# ==========================================
#
# Submodules:
#   datasets.py                — benchmark query collection
#   metrics.py                 — measurement data models
#   retrieval_evaluator.py     — retrieval quality evaluation
#   evidence_evaluator.py      — evidence quality evaluation
#
# Usage:
#     from app.evaluation.retrieval_evaluator import RetrievalEvaluator
#     from app.evaluation.evidence_evaluator import evaluate_evidence
#
#     evaluator = RetrievalEvaluator()
#     report = evaluator.evaluate_benchmark()
#
#     ev_result = evaluate_evidence({"facts": [...]})
#     print(ev_result.support_ratio)
#
# ==========================================

from app.evaluation.retrieval_evaluator import RetrievalEvaluator
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
    compute_jaccard_similarity,
    compute_stability,
)
from app.evaluation.evidence_evaluator import (
    evaluate_evidence,
    EvidenceEvaluationResult,
    EvidenceFactResult,
)

__all__ = [
    "RetrievalEvaluator",
    "DEFAULT_BENCHMARK_QUERIES",
    "BENCHMARK_FAILURE_QUERIES",
    "BenchmarkQuery",
    "RetrievalMetrics",
    "ChunkScore",
    "EvaluationResult",
    "BenchmarkReport",
    "FailedQuery",
    "compute_jaccard_similarity",
    "compute_stability",
    "evaluate_evidence",
    "EvidenceEvaluationResult",
    "EvidenceFactResult",
]
