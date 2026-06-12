# ==========================================
# EVIDENCE EVALUATION FRAMEWORK
# ==========================================
#
# Tools for measuring evidence quality and
# grounding quality without modifying the
# evidence generation pipeline.
#
# This framework exists to answer:
#   "How well grounded is our evidence pipeline?"
#
# It evaluates outputs produced by the Evidence
# Service and Factcheck Agent, measuring metrics
# such as support ratio, confidence distribution,
# source diversity, and evidence coverage.
#
# The evaluator tolerates multiple input schemas
# including:
#
#   Example A — flat facts with evidence:
#       {
#           "facts": [
#               {
#                   "fact": "...",
#                   "confidence": 0.91,
#                   "evidence": [{"url": "...", "chunk_id": ..., "text": "..."}]
#               }
#           ]
#       }
#
#   Example B — categorized facts (factcheck output):
#       {
#           "confirmed_facts": [
#               {"statement": "...", "confidence": 0.84, "supporting_chunks": [...]}
#           ]
#       }
#
#   Example C — malformed / missing fields / empty arrays
#       => all handled gracefully (never raises)
#
# Usage:
#     from app.evaluation.evidence_evaluator import evaluate_evidence, EvidenceEvaluationResult
#
#     result = evaluate_evidence({"facts": [...]})
#     print(result.support_ratio, result.average_fact_confidence)
#
# ==========================================

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from app.utils.logger import logger


# ==========================================
# DATA MODELS
# ==========================================


@dataclass(frozen=True)
class EvidenceFactResult:
    """Evaluation result for a single fact.

    Attributes:
        fact_text:      The extracted fact text (truncated for readability).
        confidence:     The fact's confidence score (0.0 if missing).
        evidence_count: Number of evidence items attached to this fact.
        supported:      Whether the fact has at least one evidence item.
        unique_sources: Number of unique source URLs among evidence.
        unique_domains: Number of unique domains among evidence.
    """

    fact_text: str = ""
    confidence: float = 0.0
    evidence_count: int = 0
    supported: bool = False
    unique_sources: int = 0
    unique_domains: int = 0


@dataclass(frozen=True)
class EvidenceEvaluationResult:
    """Complete evidence evaluation result for a single evidence output.

    All metrics are computed from the extracted facts and evidence
    without modifying the input data.

    Attributes:
        total_facts:                Number of facts extracted from the
                                    evidence output.
        supported_fact_count:       Facts with at least one supporting
                                    evidence item.
        unsupported_fact_count:     Facts without any evidence.
        average_fact_confidence:    Mean confidence across all facts
                                    (missing confidence=>0.0).
        citation_count:             Total number of evidence items across
                                    all facts.
        unique_source_count:        Number of unique evidence URLs.
        unique_domain_count:        Number of unique evidence domains
                                    (via urlparse).
        support_ratio:              supported_fact_count / total_facts
                                    (0.0 if total_facts==0).
        evidence_per_fact:  Mean number of evidence items per fact.
        coverage_score:             supported_fact_count / max(citation_count, 1)
                                    Measures how efficiently evidence is
                                    being used per supported fact.
        confidence_distribution:    Counts per confidence bucket
                                    (0.00-0.20, 0.21-0.40, 0.41-0.60,
                                     0.61-0.80, 0.81-1.00).
        per_fact_results:           Individual EvidenceFactResult for each
                                    extracted fact.
    """

    total_facts: int = 0
    supported_fact_count: int = 0
    unsupported_fact_count: int = 0
    average_fact_confidence: float = 0.0
    citation_count: int = 0
    unique_source_count: int = 0
    unique_domain_count: int = 0
    support_ratio: float = 0.0
    evidence_per_fact: float = 0.0
    coverage_score: float = 0.0
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    per_fact_results: List[EvidenceFactResult] = field(default_factory=list)


# ==========================================
# HELPER FUNCTIONS
# ==========================================


def _extract_domain(url: str) -> str:
    """Extract the registered domain from a URL.

    Args:
        url: A full URL string (e.g. ``\"https://news.cnn.com/article\"``).

    Returns:
        The netloc/domain portion (e.g. ``\"news.cnn.com\"``), or the
        original string if parsing fails.
    """
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


def _extract_fact_text(entry: Dict[str, Any]) -> str:
    """Extract fact text from a fact entry dict.

    Tries known keys in order: ``\"fact\"``, ``\"statement\"``.
    Returns empty string if none found.

    Args:
        entry: A fact entry dict from the evidence output.

    Returns:
        The extracted fact text, or ``\"\"`` if unavailable.
    """
    for key in ("fact", "statement"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_confidence(entry: Dict[str, Any]) -> float:
    """Extract confidence score from a fact entry.

    Args:
        entry: A fact entry dict.

    Returns:
        Confidence as a float in [0, 1], or 0.0 if missing/invalid.
    """
    value = entry.get("confidence")
    if isinstance(value, (int, float)):
        return float(max(0.0, min(1.0, value)))
    return 0.0


def _extract_evidence_list(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the list of evidence items from a fact entry.

    Tries known keys in order: ``\"evidence\"``, ``\"supporting_chunks\"``.

    Args:
        entry: A fact entry dict.

    Returns:
        List of evidence item dicts (may be empty). Invalid items
        (non-dict) are filtered out.
    """
    for key in ("evidence", "supporting_chunks"):
        raw = entry.get(key)
        if isinstance(raw, list):
            # Filter to only dict items
            return [item for item in raw if isinstance(item, dict)]
    return []


def _extract_evidence_url(evidence_item: Dict[str, Any]) -> str:
    """Extract the URL from a single evidence item.

    Args:
        evidence_item: An evidence item dict.

    Returns:
        The URL string, or ``\"\"`` if unavailable.
    """
    url = evidence_item.get("url", evidence_item.get("source_url", ""))
    if isinstance(url, str):
        return url
    return ""


def _compute_confidence_distribution(
    confidences: List[float],
) -> Dict[str, int]:
    """Bucket confidence values into predefined ranges.

    Buckets::

        0.00-0.20, 0.21-0.40, 0.41-0.60, 0.61-0.80, 0.81-1.00

    Args:
        confidences: List of confidence values in [0, 1].

    Returns:
        Dict mapping bucket label (e.g. ``\"0.81-1.00\"``) to count.
    """
    distribution: Dict[str, int] = {
        "0.00-0.20": 0,
        "0.21-0.40": 0,
        "0.41-0.60": 0,
        "0.61-0.80": 0,
        "0.81-1.00": 0,
    }

    for c in confidences:
        if c <= 0.20:
            distribution["0.00-0.20"] += 1
        elif c <= 0.40:
            distribution["0.21-0.40"] += 1
        elif c <= 0.60:
            distribution["0.41-0.60"] += 1
        elif c <= 0.80:
            distribution["0.61-0.80"] += 1
        else:
            distribution["0.81-1.00"] += 1

    return distribution


# ==========================================
# FACT EXTRACTION
# ==========================================


def _extract_facts(
    evidence_output: Any,
) -> List[Dict[str, Any]]:
    """Extract all fact entries from an evidence output.

    Handles multiple top-level schemas:

    * ``\"facts\"`` — flat list of fact dicts
    * ``\"confirmed_facts\"``, ``\"disputed_facts\"``, ``\"low_confidence_facts\"``
      — categorized lists (all merged into one flat list)

    Non-dict items and non-dict top-level inputs are silently skipped.

    Args:
        evidence_output: Raw evidence output (any type). Expected to be
                         a dict with one of the supported top-level keys.

    Returns:
        Flat list of fact entry dicts extracted from the output.
    """
    if not isinstance(evidence_output, dict):
        return []

    facts: List[Dict[str, Any]] = []

    # --- Try "facts" key (flat list) ---
    raw_facts = evidence_output.get("facts")
    if isinstance(raw_facts, list):
        for item in raw_facts:
            if isinstance(item, dict):
                facts.append(item)

    # --- Try categorized keys ---
    for category_key in (
        "confirmed_facts",
        "disputed_facts",
        "low_confidence_facts",
    ):
        raw_category = evidence_output.get(category_key)
        if isinstance(raw_category, list):
            for item in raw_category:
                if isinstance(item, dict):
                    # Avoid duplicates if already extracted via "facts"
                    if item not in facts:
                        facts.append(item)

    return facts


# ==========================================
# MAIN EVALUATION FUNCTION
# ==========================================


def evaluate_evidence(
    evidence_output: Any,
) -> EvidenceEvaluationResult:
    """Evaluate the quality and grounding of an evidence output.

    This is the main entry point for the Evidence Evaluation Framework.
    It accepts any input structure (dict, list, None, etc.) and returns
    a structured EvidenceEvaluationResult with all computed metrics.

    The function **never raises exceptions**. All malformed inputs are
    handled via graceful degradation (zeroed results on extreme failure).

    Args:
        evidence_output: Raw evidence output from the Evidence Service
                         or Factcheck Agent. Expected to be a dict
                         containing fact entries under known keys.

    Returns:
        An EvidenceEvaluationResult with all metrics computed. Returns
        a zeroed result if the input is unparseable.

    Example::

        result = evaluate_evidence({
            "facts": [
                {"fact": "...", "confidence": 0.9, "evidence": [{"url": "..."}]},
                {"fact": "...", "confidence": 0.5, "evidence": []},
            ]
        })
        print(result.support_ratio)  # 0.5
    """
    logger.info("[EVIDENCE EVAL] Starting evidence evaluation")

    try:
        # ----------------------------------------------------------
        # 1. Extract facts
        # ----------------------------------------------------------
        raw_entries = _extract_facts(evidence_output)
        total_facts = len(raw_entries)

        if total_facts == 0:
            logger.info(
                "[EVIDENCE EVAL] No facts extracted, returning zeroed result"
            )
            return EvidenceEvaluationResult(
                total_facts=0,
                confidence_distribution=_compute_confidence_distribution([]),
            )

        # ----------------------------------------------------------
        # 2. Process each fact entry
        # ----------------------------------------------------------
        per_fact_results: List[EvidenceFactResult] = []
        all_confidences: List[float] = []
        all_unique_urls: set[str] = set()
        all_unique_domains: set[str] = set()
        total_citations = 0
        supported_count = 0

        for entry in raw_entries:
            fact_text = _extract_fact_text(entry)
            confidence = _extract_confidence(entry)
            evidence_items = _extract_evidence_list(entry)
            evidence_count = len(evidence_items)

            all_confidences.append(confidence)

            # --- Collect evidence URLs ---
            fact_urls: set[str] = set()
            fact_domains: set[str] = set()
            for ev in evidence_items:
                ev_url = _extract_evidence_url(ev)
                if ev_url:
                    all_unique_urls.add(ev_url)
                    fact_urls.add(ev_url)
                    domain = _extract_domain(ev_url)
                    all_unique_domains.add(domain)
                    fact_domains.add(domain)

            is_supported = evidence_count > 0
            if is_supported:
                supported_count += 1

            total_citations += evidence_count

            per_fact_results.append(
                EvidenceFactResult(
                    fact_text=fact_text[:200],
                    confidence=confidence,
                    evidence_count=evidence_count,
                    supported=is_supported,
                    unique_sources=len(fact_urls),
                    unique_domains=len(fact_domains),
                )
            )

        # ----------------------------------------------------------
        # 3. Compute aggregate metrics
        # ----------------------------------------------------------
        unsupported_count = total_facts - supported_count
        support_ratio = (
            round(supported_count / total_facts, 4) if total_facts > 0 else 0.0
        )
        avg_confidence = (
            round(statistics.mean(all_confidences), 4) if all_confidences else 0.0
        )
        avg_evidence_per_fact = (
            round(total_citations / total_facts, 4) if total_facts > 0 else 0.0
        )
        coverage_score = (
            round(supported_count / max(total_citations, 1), 4)
            if total_citations > 0
            else 0.0
        )
        confidence_distribution = _compute_confidence_distribution(all_confidences)

        result = EvidenceEvaluationResult(
            total_facts=total_facts,
            supported_fact_count=supported_count,
            unsupported_fact_count=unsupported_count,
            average_fact_confidence=avg_confidence,
            citation_count=total_citations,
            unique_source_count=len(all_unique_urls),
            unique_domain_count=len(all_unique_domains),
            support_ratio=support_ratio,
            evidence_per_fact=avg_evidence_per_fact,
            coverage_score=coverage_score,
            confidence_distribution=confidence_distribution,
            per_fact_results=per_fact_results,
        )

        logger.info(
            "[EVIDENCE EVAL] Complete: %d facts, %d supported, "
            "%.2f%% support ratio, %.4f avg confidence, "
            "%d citations, %d unique sources",
            total_facts,
            supported_count,
            support_ratio * 100,
            avg_confidence,
            total_citations,
            len(all_unique_urls),
        )

        return result

    except Exception as e:
        logger.error(
            "[EVIDENCE EVAL] Unexpected failure: %s. Returning zeroed result.",
            e,
        )
        return EvidenceEvaluationResult(
            total_facts=0,
            confidence_distribution=_compute_confidence_distribution([]),
        )
