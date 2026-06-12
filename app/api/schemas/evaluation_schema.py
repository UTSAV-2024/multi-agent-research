# ==========================================
# EVALUATION API SCHEMAS
# ==========================================
#
# Request/response models for the evaluation
# endpoints that expose the evaluation
# frameworks via the public API.
#
# ==========================================

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==========================================
# RETRIEVAL EVALUATION
# ==========================================


class RetrievalEvaluationRequest(BaseModel):
    """Request payload for POST /api/v1/evaluate/retrieval."""

    queries: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of search queries to evaluate. At least 1, at most 100.",
        examples=[["Operation Barbarossa", "CRISPR gene editing"]],
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top results to retrieve per query (1–20).",
    )


# ==========================================
# EVIDENCE EVALUATION
# ==========================================


class EvidenceEvaluationRequest(BaseModel):
    """Request payload for POST /api/v1/evaluate/evidence."""

    evidence: Any = Field(
        ...,
        description=(
            "Evidence output to evaluate. Accepts the same formats as "
            "the evaluate_evidence() function: a dict with 'facts', "
            "'confirmed_facts', 'disputed_facts', 'low_confidence_facts' "
            "keys, or alternative schemas with 'statement' and "
            "'supporting_chunks' fields."
        ),
        examples=[
            {
                "facts": [
                    {
                        "fact": "mRNA vaccines teach cells to produce a protein",
                        "confidence": 0.91,
                        "evidence": [{"url": "https://example.com/article", "chunk_id": 1}],
                    }
                ]
            }
        ],
    )
