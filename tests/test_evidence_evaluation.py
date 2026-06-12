# ==========================================
# EVIDENCE EVALUATION FRAMEWORK — TESTS
# ==========================================
#
# Tests for the evidence evaluator covering:
#   - All facts supported
#   - Partial support
#   - No evidence
#   - Empty input
#   - Malformed evidence structures
#   - Missing confidence
#   - Missing URLs
#   - Multiple domains
#   - Confidence bucketing
#   - Zero facts
#   - Evidence arrays with invalid entries
#   - Deterministic outputs
#   - Graceful degradation
#   - Large evidence inputs
#   - Mixed schema support (facts vs confirmed_facts)
#   - Coverage score
#   - Helper functions
#
# Run with:
#     pytest tests/test_evidence_evaluation.py -v
#
# ==========================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.evaluation.evidence_evaluator import (
    evaluate_evidence,
    EvidenceEvaluationResult,
    EvidenceFactResult,
    _extract_facts,
    _extract_fact_text,
    _extract_confidence,
    _extract_evidence_list,
    _extract_evidence_url,
    _extract_domain,
    _compute_confidence_distribution,
)


# ==========================================
# HELPER FUNCTION TESTS
# ==========================================


class TestExtractDomain:
    """Tests for _extract_domain()."""

    def test_standard_url(self):
        """Standard URL extracts netloc."""
        assert _extract_domain("https://cnn.com/article") == "cnn.com"

    def test_url_with_subdomain(self):
        """URL with subdomain extracts full netloc."""
        assert (
            _extract_domain("https://news.bbc.co.uk/story")
            == "news.bbc.co.uk"
        )

    def test_empty_url(self):
        """Empty URL returns empty string."""
        assert _extract_domain("") == ""

    def test_invalid_url(self):
        """Invalid URL returns the string itself or empty."""
        result = _extract_domain("not-a-url")
        assert result == "not-a-url" or result == ""


class TestExtractFactText:
    """Tests for _extract_fact_text()."""

    def test_fact_key(self):
        """Extracts from 'fact' key."""
        result = _extract_fact_text({"fact": "The sky is blue"})
        assert result == "The sky is blue"

    def test_statement_key(self):
        """Extracts from 'statement' key when 'fact' is absent."""
        result = _extract_fact_text({"statement": "The earth is round"})
        assert result == "The earth is round"

    def test_fact_preferred_over_statement(self):
        """'fact' key is preferred over 'statement'."""
        result = _extract_fact_text({
            "fact": "fact value",
            "statement": "statement value",
        })
        assert result == "fact value"

    def test_missing_fact_text(self):
        """Returns empty string when neither key is present."""
        result = _extract_fact_text({"confidence": 0.9})
        assert result == ""

    def test_non_string_fact(self):
        """Non-string fact value returns empty string."""
        result = _extract_fact_text({"fact": 123})
        assert result == ""

    def test_empty_entry(self):
        """Empty dict returns empty string."""
        result = _extract_fact_text({})
        assert result == ""


class TestExtractConfidence:
    """Tests for _extract_confidence()."""

    def test_valid_confidence(self):
        """Extracts confidence from dict."""
        assert _extract_confidence({"confidence": 0.85}) == 0.85

    def test_missing_confidence_defaults_zero(self):
        """Missing confidence returns 0.0."""
        assert _extract_confidence({"fact": "..."}) == 0.0

    def test_none_confidence(self):
        """None confidence returns 0.0."""
        assert _extract_confidence({"confidence": None}) == 0.0

    def test_clamps_above_one(self):
        """Confidence > 1.0 is clamped to 1.0."""
        assert _extract_confidence({"confidence": 1.5}) == 1.0

    def test_clamps_below_zero(self):
        """Confidence < 0.0 is clamped to 0.0."""
        assert _extract_confidence({"confidence": -0.5}) == 0.0

    def test_string_confidence(self):
        """String confidence returns 0.0."""
        assert _extract_confidence({"confidence": "high"}) == 0.0


class TestExtractEvidenceList:
    """Tests for _extract_evidence_list()."""

    def test_evidence_key(self):
        """Extracts from 'evidence' key."""
        items = [{"url": "https://a.com"}]
        result = _extract_evidence_list({"evidence": items})
        assert result == items

    def test_supporting_chunks_key(self):
        """Extracts from 'supporting_chunks' key."""
        items = [{"url": "https://b.com"}]
        result = _extract_evidence_list({
            "supporting_chunks": items,
        })
        assert result == items

    def test_evidence_preferred(self):
        """'evidence' key is preferred over 'supporting_chunks'."""
        result = _extract_evidence_list({
            "evidence": [{"url": "https://a.com"}],
            "supporting_chunks": [{"url": "https://b.com"}],
        })
        assert len(result) == 1
        assert result[0]["url"] == "https://a.com"

    def test_missing_evidence(self):
        """Missing evidence key returns empty list."""
        result = _extract_evidence_list({"fact": "..."})
        assert result == []

    def test_non_list_evidence(self):
        """Non-list evidence returns empty list."""
        result = _extract_evidence_list({"evidence": "not a list"})
        assert result == []

    def test_filters_non_dict_items(self):
        """Non-dict items in evidence list are filtered out."""
        result = _extract_evidence_list({
            "evidence": [
                {"url": "https://a.com"},
                "not a dict",
                None,
                123,
                {"url": "https://b.com"},
            ],
        })
        assert len(result) == 2
        assert result[0]["url"] == "https://a.com"
        assert result[1]["url"] == "https://b.com"

    def test_empty_evidence_list(self):
        """Empty evidence list returns empty list."""
        result = _extract_evidence_list({"evidence": []})
        assert result == []


class TestExtractEvidenceUrl:
    """Tests for _extract_evidence_url()."""

    def test_url_key(self):
        """Extracts from 'url' key."""
        assert _extract_evidence_url({"url": "https://a.com"}) == "https://a.com"

    def test_source_url_key(self):
        """Extracts from 'source_url' key."""
        assert _extract_evidence_url({
            "source_url": "https://b.com",
        }) == "https://b.com"

    def test_url_preferred(self):
        """'url' key is preferred over 'source_url'."""
        result = _extract_evidence_url({
            "url": "https://preferred.com",
            "source_url": "https://fallback.com",
        })
        assert result == "https://preferred.com"

    def test_missing_url(self):
        """Missing URL returns empty string."""
        assert _extract_evidence_url({"chunk_id": 1}) == ""

    def test_non_string_url(self):
        """Non-string URL returns empty string."""
        assert _extract_evidence_url({"url": 123}) == ""


class TestExtractFacts:
    """Tests for _extract_facts()."""

    def test_facts_key(self):
        """Extracts from 'facts' key."""
        data = {"facts": [{"fact": "a"}, {"fact": "b"}]}
        result = _extract_facts(data)
        assert len(result) == 2

    def test_confirmed_facts_key(self):
        """Extracts from 'confirmed_facts' key."""
        data = {"confirmed_facts": [{"statement": "a"}]}
        result = _extract_facts(data)
        assert len(result) == 1

    def test_all_categorized_keys(self):
        """Extracts from all categorized keys."""
        data = {
            "confirmed_facts": [{"fact": "a"}],
            "disputed_facts": [{"fact": "b"}],
            "low_confidence_facts": [{"fact": "c"}],
        }
        result = _extract_facts(data)
        assert len(result) == 3

    def test_non_dict_input(self):
        """Non-dict input returns empty list."""
        assert _extract_facts(None) == []
        assert _extract_facts("string") == []
        assert _extract_facts(123) == []
        assert _extract_facts([1, 2, 3]) == []

    def test_filters_non_dict_items(self):
        """Non-dict items within fact lists are filtered."""
        data = {"facts": [{"fact": "a"}, "not a dict", None, {"fact": "b"}]}
        result = _extract_facts(data)
        assert len(result) == 2

    def test_empty_input_dict(self):
        """Empty dict returns empty list."""
        result = _extract_facts({})
        assert result == []

    def test_does_not_duplicate(self):
        """Facts from 'facts' key are not duplicated in categorized."""
        data = {
            "facts": [{"fact": "a"}],
            "confirmed_facts": [{"fact": "a"}],
        }
        result = _extract_facts(data)
        # "a" appears in both, but should only be counted once
        assert len(result) == 1


class TestConfidenceDistribution:
    """Tests for _compute_confidence_distribution()."""

    def test_all_buckets_filled(self):
        """Confidences populate the correct buckets."""
        confidences = [0.05, 0.25, 0.45, 0.65, 0.95]
        dist = _compute_confidence_distribution(confidences)
        assert dist["0.00-0.20"] == 1
        assert dist["0.21-0.40"] == 1
        assert dist["0.41-0.60"] == 1
        assert dist["0.61-0.80"] == 1
        assert dist["0.81-1.00"] == 1

    def test_boundary_values(self):
        """Boundary values go to the correct buckets."""
        confidences = [0.20, 0.21, 0.40, 0.41, 0.60, 0.61, 0.80, 0.81]
        dist = _compute_confidence_distribution(confidences)
        assert dist["0.00-0.20"] == 1
        assert dist["0.21-0.40"] == 2
        assert dist["0.41-0.60"] == 2
        assert dist["0.61-0.80"] == 2
        assert dist["0.81-1.00"] == 1

    def test_empty_list(self):
        """Empty list returns all-zero distribution."""
        dist = _compute_confidence_distribution([])
        assert all(v == 0 for v in dist.values())
        assert len(dist) == 5

    def test_all_high_confidence(self):
        """All high confidences go to top bucket."""
        confidences = [0.95, 0.98, 1.0]
        dist = _compute_confidence_distribution(confidences)
        assert dist["0.81-1.00"] == 3
        assert sum(dist.values()) == 3


# ==========================================
# EVIDENCE EVALUATION — CORE ASSERTION TESTS
# ==========================================


class TestAllFactsSupported:
    """Scenario: All facts have evidence."""

    def test_all_supported(self):
        """All facts with evidence yield support_ratio=1.0."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Fact one",
                    "confidence": 0.9,
                    "evidence": [{"url": "https://a.com"}, {"url": "https://b.com"}],
                },
                {
                    "fact": "Fact two",
                    "confidence": 0.8,
                    "evidence": [{"url": "https://c.com"}],
                },
            ],
        })
        assert result.total_facts == 2
        assert result.supported_fact_count == 2
        assert result.unsupported_fact_count == 0
        assert result.support_ratio == 1.0
        assert result.citation_count == 3


class TestPartialSupport:
    """Scenario: Some facts have evidence, some don't."""

    def test_partial_support(self):
        """Half-supported facts yield support_ratio=0.5."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Supported fact",
                    "confidence": 0.9,
                    "evidence": [{"url": "https://a.com"}],
                },
                {
                    "fact": "Unsupported fact",
                    "confidence": 0.5,
                    "evidence": [],
                },
            ],
        })
        assert result.total_facts == 2
        assert result.supported_fact_count == 1
        assert result.unsupported_fact_count == 1
        assert result.support_ratio == 0.5


class TestNoEvidence:
    """Scenario: No fact has evidence."""

    def test_no_evidence_at_all(self):
        """No evidence yields support_ratio=0.0."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "Fact a", "confidence": 0.7, "evidence": []},
                {"fact": "Fact b", "confidence": 0.6, "evidence": []},
            ],
        })
        assert result.total_facts == 2
        assert result.supported_fact_count == 0
        assert result.unsupported_fact_count == 2
        assert result.support_ratio == 0.0
        assert result.citation_count == 0


class TestEmptyInput:
    """Scenario: Empty or unparseable inputs."""

    def test_empty_dict(self):
        """Empty dict yields zeroed result."""
        result = evaluate_evidence({})
        assert result.total_facts == 0
        assert result.support_ratio == 0.0
        assert result.citation_count == 0

    def test_none_input(self):
        """None input yields zeroed result."""
        result = evaluate_evidence(None)
        assert result.total_facts == 0
        assert result.support_ratio == 0.0

    def test_string_input(self):
        """String input yields zeroed result."""
        result = evaluate_evidence("invalid")
        assert result.total_facts == 0
        assert result.support_ratio == 0.0

    def test_list_input(self):
        """List input yields zeroed result."""
        result = evaluate_evidence([1, 2, 3])
        assert result.total_facts == 0

    def test_no_facts_key(self):
        """Dict without any known facts key yields zeroed result."""
        result = evaluate_evidence({"unrelated_key": "value"})
        assert result.total_facts == 0


class TestMalformedStructures:
    """Scenario: Malformed evidence structures."""

    def test_non_dict_in_facts(self):
        """Non-dict items in facts list are skipped."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "Valid fact", "evidence": [{"url": "https://a.com"}]},
                "not a dict",
                None,
                123,
            ],
        })
        assert result.total_facts == 1
        assert result.supported_fact_count == 1

    def test_null_fact_entry(self):
        """Null fact entries are skipped."""
        result = evaluate_evidence({
            "facts": [
                None,
                {"fact": "Real fact", "evidence": [{"url": "https://a.com"}]},
                None,
            ],
        })
        assert result.total_facts == 1
        assert result.supported_fact_count == 1

    def test_empty_fact_dict(self):
        """Empty fact dict is counted (no fact text, no evidence)."""
        result = evaluate_evidence({
            "facts": [
                {},
                {"fact": "Real fact", "evidence": [{"url": "https://a.com"}]},
            ],
        })
        assert result.total_facts == 2
        assert result.supported_fact_count == 1
        assert result.unsupported_fact_count == 1

    def test_evidence_with_non_dict_items(self):
        """Evidence arrays with mixed valid/invalid items."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Fact with mixed evidence",
                    "confidence": 0.8,
                    "evidence": [
                        {"url": "https://valid.com"},
                        "not a dict",
                        None,
                        {"url": "https://also-valid.com"},
                    ],
                },
            ],
        })
        assert result.total_facts == 1
        assert result.supported_fact_count == 1
        assert result.citation_count == 2
        assert result.unique_source_count == 2


class TestMissingConfidence:
    """Scenario: Missing or invalid confidence values."""

    def test_missing_confidence_defaults_zero(self):
        """Missing confidence is treated as 0.0."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "No confidence", "evidence": []},
            ],
        })
        assert result.average_fact_confidence == 0.0
        assert result.confidence_distribution["0.00-0.20"] == 1

    def test_mixed_confidence_handling(self):
        """Mixed present/missing confidence values are averaged correctly."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "Has confidence", "confidence": 0.8, "evidence": []},
                {"fact": "No confidence", "evidence": []},
            ],
        })
        assert result.average_fact_confidence == 0.4  # (0.8 + 0.0) / 2

    def test_all_missing_confidence(self):
        """All missing confidence yields average of 0.0."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "A", "evidence": []},
                {"fact": "B", "evidence": []},
                {"fact": "C", "evidence": []},
            ],
        })
        assert result.average_fact_confidence == 0.0
        assert result.confidence_distribution["0.00-0.20"] == 3

    def test_none_confidence_treated_as_zero(self):
        """None confidence is treated as 0.0."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "A", "confidence": None, "evidence": []},
            ],
        })
        assert result.average_fact_confidence == 0.0


class TestMissingUrls:
    """Scenario: Evidence items without URLs."""

    def test_evidence_without_urls(self):
        """Evidence items without URLs don't contribute to source counts."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Fact with URL-less evidence",
                    "confidence": 0.9,
                    "evidence": [
                        {"chunk_id": 1},
                        {"chunk_id": 2, "url": ""},
                        {"chunk_id": 3, "url": "https://real.com"},
                    ],
                },
            ],
        })
        assert result.total_facts == 1
        assert result.citation_count == 3
        assert result.unique_source_count == 1
        assert result.unique_domain_count == 1

    def test_all_evidence_without_urls(self):
        """All evidence without URLs yields zero source counts."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Fact",
                    "confidence": 0.8,
                    "evidence": [
                        {"chunk_id": 1},
                        {"chunk_id": 2},
                    ],
                },
            ],
        })
        assert result.citation_count == 2
        assert result.unique_source_count == 0
        assert result.unique_domain_count == 0


class TestMultipleDomains:
    """Scenario: Evidence from multiple domains."""

    def test_multiple_domains(self):
        """Multiple domains are counted correctly."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Fact with diverse sources",
                    "confidence": 0.9,
                    "evidence": [
                        {"url": "https://cnn.com/article-a"},
                        {"url": "https://cnn.com/article-b"},
                        {"url": "https://reuters.com/article-c"},
                        {"url": "https://bbc.co.uk/article-d"},
                    ],
                },
            ],
        })
        assert result.unique_source_count == 4
        assert result.unique_domain_count == 3  # cnn.com, reuters.com, bbc.co.uk

    def test_same_domain_multiple_urls(self):
        """Same domain with multiple URLs counts as one domain."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Single domain fact",
                    "confidence": 0.7,
                    "evidence": [
                        {"url": "https://cnn.com/a"},
                        {"url": "https://cnn.com/b"},
                        {"url": "https://cnn.com/c"},
                    ],
                },
            ],
        })
        assert result.unique_source_count == 3
        assert result.unique_domain_count == 1


class TestConfidenceBucketing:
    """Tests for confidence distribution bucketing."""

    def test_confidence_buckets_correct(self):
        """Confidences are bucketed correctly."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "A", "confidence": 0.05, "evidence": []},
                {"fact": "B", "confidence": 0.25, "evidence": []},
                {"fact": "C", "confidence": 0.45, "evidence": []},
                {"fact": "D", "confidence": 0.65, "evidence": []},
                {"fact": "E", "confidence": 0.95, "evidence": []},
                {"fact": "F", "confidence": 0.15, "evidence": []},
                {"fact": "G", "confidence": 0.85, "evidence": []},
            ],
        })
        dist = result.confidence_distribution
        assert dist["0.00-0.20"] == 2
        assert dist["0.21-0.40"] == 1
        assert dist["0.41-0.60"] == 1
        assert dist["0.61-0.80"] == 1
        assert dist["0.81-1.00"] == 2
        assert sum(dist.values()) == 7

    def test_zero_facts_buckets_all_zero(self):
        """Zero facts yields all-zero buckets."""
        result = evaluate_evidence({})
        dist = result.confidence_distribution
        assert all(v == 0 for v in dist.values())


class TestCoverageScore:
    """Tests for the coverage_score metric."""

    def test_coverage_score_perfect(self):
        """Perfect coverage: each fact has exactly one evidence item."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "A",
                    "evidence": [{"url": "https://a.com"}],
                },
                {
                    "fact": "B",
                    "evidence": [{"url": "https://b.com"}],
                },
            ],
        })
        # coverage = supported / max(citations, 1) = 2 / 2 = 1.0
        assert result.coverage_score == 1.0

    def test_coverage_score_half(self):
        """Moderate coverage: 10 facts, 20 citations -> 0.5."""
        facts = [
            {
                "fact": f"Fact {i}",
                "evidence": [{"url": f"https://a{i}.com"}, {"url": f"https://b{i}.com"}],
            }
            for i in range(10)
        ]
        result = evaluate_evidence({"facts": facts})
        # coverage = 10 / 20 = 0.5
        assert result.coverage_score == 0.5
        assert result.supported_fact_count == 10
        assert result.citation_count == 20

    def test_coverage_score_zero_supported(self):
        """Zero supported facts yields coverage_score=0.0."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "A", "evidence": []},
                {"fact": "B", "evidence": []},
            ],
        })
        assert result.coverage_score == 0.0

    def test_coverage_score_zero_citations(self):
        """Zero citations yields coverage_score=0.0."""
        result = evaluate_evidence({
            "facts": [],
        })
        assert result.coverage_score == 0.0

    def test_coverage_score_no_facts(self):
        """No facts yields coverage_score=0.0."""
        result = evaluate_evidence({})
        assert result.coverage_score == 0.0


class TestEvidencePerFact:
    """Tests for evidence_per_fact."""

    def test_evidence_per_fact_value(self):
        """evidence_per_fact is calculated correctly."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "A",
                    "evidence": [{"url": "https://a.com"}, {"url": "https://b.com"}],
                },
                {
                    "fact": "B",
                    "evidence": [{"url": "https://c.com"}],
                },
            ],
        })
        assert result.evidence_per_fact == 1.5  # 3 / 2

    def test_evidence_per_fact_zero_facts(self):
        """Zero facts yields evidence_per_fact=0.0."""
        result = evaluate_evidence({})
        assert result.evidence_per_fact == 0.0


class TestSupportRatio:
    """Tests for support_ratio edge cases."""

    def test_support_ratio_zero_facts(self):
        """Zero facts yields support_ratio=0.0."""
        result = evaluate_evidence({})
        assert result.support_ratio == 0.0

    def test_support_ratio_all_unsupported(self):
        """All unsupported yields support_ratio=0.0."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "A", "evidence": []},
                {"fact": "B", "evidence": []},
            ],
        })
        assert result.support_ratio == 0.0


class TestDeterministicOutput:
    """Scenario: Same input always produces same output."""

    def test_deterministic(self):
        """Same input produces identical results."""
        input_data = {
            "facts": [
                {
                    "fact": "Deterministic fact",
                    "confidence": 0.85,
                    "evidence": [
                        {"url": "https://a.com", "chunk_id": 1},
                        {"url": "https://b.com", "chunk_id": 2},
                    ],
                },
                {
                    "fact": "Unsupported fact",
                    "confidence": 0.3,
                    "evidence": [],
                },
            ],
        }
        r1 = evaluate_evidence(input_data)
        r2 = evaluate_evidence(input_data)

        assert r1.total_facts == r2.total_facts
        assert r1.supported_fact_count == r2.supported_fact_count
        assert r1.support_ratio == r2.support_ratio
        assert r1.average_fact_confidence == r2.average_fact_confidence
        assert r1.citation_count == r2.citation_count
        assert r1.unique_source_count == r2.unique_source_count
        assert r1.unique_domain_count == r2.unique_domain_count
        assert r1.coverage_score == r2.coverage_score
        assert r1.confidence_distribution == r2.confidence_distribution


class TestGracefulDegradation:
    """Scenario: Evaluation never raises exceptions."""

    def test_none_raises_no_exception(self):
        """None input does not raise."""
        try:
            result = evaluate_evidence(None)
            assert result.total_facts == 0
        except Exception:
            pytest.fail("evaluate_evidence raised on None input")

    def test_malformed_raises_no_exception(self):
        """Malformed input does not raise."""
        try:
            result = evaluate_evidence({
                "facts": [None, "bad", {}, {"evidence": "not a list"}],
            })
            assert result.total_facts >= 0
        except Exception:
            pytest.fail("evaluate_evidence raised on malformed input")

    def test_list_raises_no_exception(self):
        """List input does not raise."""
        try:
            result = evaluate_evidence([1, 2, 3])
            assert result.total_facts == 0
        except Exception:
            pytest.fail("evaluate_evidence raised on list input")

    def test_exception_inside_fact_processing(self):
        """Unexpected structure inside facts does not crash."""
        try:
            # This shouldn't happen normally, but test the safety net
            class WeirdDict(dict):
                def get(self, key, default=None):
                    raise RuntimeError("Simulated failure")
            result = evaluate_evidence({
                "facts": [WeirdDict(fact="weird")],
            })
            assert result.total_facts == 0
        except Exception:
            pytest.fail("evaluate_evidence raised on exceptional fact entry")


class TestLargeEvidenceInput:
    """Scenario: Large evidence inputs."""

    def test_large_input_performance(self):
        """Many facts and evidence items are handled without error."""
        facts = []
        for i in range(100):
            evidence = [
                {"url": f"https://source{j}.com/article-{i}"}
                for j in range(5)
            ]
            facts.append({
                "fact": f"Fact number {i} about topic X",
                "confidence": round(i / 100.0, 2),
                "evidence": evidence,
            })

        result = evaluate_evidence({"facts": facts})

        assert result.total_facts == 100
        assert result.supported_fact_count == 100
        assert result.citation_count == 500
        assert result.support_ratio == 1.0
        assert result.average_fact_confidence == pytest.approx(0.495, 0.01)
        assert result.unique_source_count == 500  # 100 * 5, all unique
        assert result.unique_domain_count == 5  # 5 unique domains (source0-4.com) repeated across all facts
        assert result.coverage_score == 0.2  # 100 / 500

    def test_large_input_with_unsupported(self):
        """Large input with mixed supported/unsupported facts."""
        facts = []
        for i in range(50):
            is_supported = i % 2 == 0
            facts.append({
                "fact": f"Fact {i}",
                "confidence": 0.9,
                "evidence": (
                    [{"url": f"https://a{i}.com"}] if is_supported else []
                ),
            })

        result = evaluate_evidence({"facts": facts})

        assert result.total_facts == 50
        assert result.supported_fact_count == 25
        assert result.unsupported_fact_count == 25
        assert result.support_ratio == 0.5


class TestMixedSchemaSupport:
    """Scenario: Different input schemas work correctly."""

    def test_confirmed_facts_schema(self):
        """'confirmed_facts' key works as input."""
        result = evaluate_evidence({
            "confirmed_facts": [
                {
                    "statement": "Confirmed fact",
                    "confidence": 0.84,
                    "supporting_chunks": [
                        {"url": "https://a.com", "chunk_id": 1},
                    ],
                },
            ],
        })
        assert result.total_facts == 1
        assert result.supported_fact_count == 1
        assert result.citation_count == 1
        assert result.average_fact_confidence == 0.84

    def test_disputed_facts_schema(self):
        """'disputed_facts' key works as input."""
        result = evaluate_evidence({
            "disputed_facts": [
                {
                    "statement": "Disputed claim",
                    "confidence": 0.40,
                    "evidence": [],
                },
            ],
        })
        assert result.total_facts == 1
        assert result.supported_fact_count == 0
        assert result.average_fact_confidence == 0.40

    def test_low_confidence_facts_schema(self):
        """'low_confidence_facts' key works as input."""
        result = evaluate_evidence({
            "low_confidence_facts": [
                {
                    "statement": "Weak claim",
                    "confidence": 0.25,
                    "evidence": [{"url": "https://weak.com"}],
                },
            ],
        })
        assert result.total_facts == 1
        assert result.supported_fact_count == 1
        assert result.average_fact_confidence == 0.25

    def test_mixed_all_categories(self):
        """All three categorized keys merged into one evaluation."""
        result = evaluate_evidence({
            "confirmed_facts": [
                {
                    "statement": "High confidence",
                    "confidence": 0.95,
                    "evidence": [{"url": "https://conf.com"}],
                },
            ],
            "disputed_facts": [
                {
                    "statement": "Disputed",
                    "confidence": 0.50,
                    "evidence": [],
                },
            ],
            "low_confidence_facts": [
                {
                    "statement": "Weak",
                    "confidence": 0.30,
                    "evidence": [{"url": "https://weak.com"}],
                },
            ],
        })
        assert result.total_facts == 3
        assert result.supported_fact_count == 2
        assert result.unsupported_fact_count == 1
        assert result.support_ratio == pytest.approx(2 / 3, 0.001)
        assert result.average_fact_confidence == pytest.approx(
            (0.95 + 0.50 + 0.30) / 3, 0.001
        )
        assert result.citation_count == 2
        assert result.unique_source_count == 2

    def test_evidence_supporting_chunks_fallback(self):
        """'supporting_chunks' fallback works when 'evidence' is absent."""
        result = evaluate_evidence({
            "facts": [
                {
                    "statement": "Fact with supporting_chunks",
                    "confidence": 0.75,
                    "supporting_chunks": [
                        {"url": "https://a.com"},
                        {"url": "https://b.com"},
                    ],
                },
            ],
        })
        assert result.total_facts == 1
        assert result.supported_fact_count == 1
        assert result.citation_count == 2

    def test_facts_with_source_url(self):
        """Evidence with 'source_url' key works correctly."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Fact with source_url",
                    "confidence": 0.8,
                    "evidence": [
                        {"source_url": "https://a.com", "chunk_id": 1},
                        {"source_url": "https://a.com", "chunk_id": 2},
                    ],
                },
            ],
        })
        assert result.unique_source_count == 1
        assert result.citation_count == 2


class TestPerFactResults:
    """Tests for per-fact breakdown."""

    def test_per_fact_evidence_count(self):
        """Per-fact results contain correct evidence counts."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Two evidence",
                    "confidence": 0.9,
                    "evidence": [
                        {"url": "https://a.com"},
                        {"url": "https://b.com"},
                    ],
                },
                {
                    "fact": "No evidence",
                    "confidence": 0.3,
                    "evidence": [],
                },
            ],
        })
        assert len(result.per_fact_results) == 2
        assert result.per_fact_results[0].evidence_count == 2
        assert result.per_fact_results[0].supported is True
        assert result.per_fact_results[1].evidence_count == 0
        assert result.per_fact_results[1].supported is False

    def test_per_fact_sources(self):
        """Per-fact results contain correct source counts."""
        result = evaluate_evidence({
            "facts": [
                {
                    "fact": "Diverse evidence",
                    "confidence": 0.8,
                    "evidence": [
                        {"url": "https://cnn.com/a"},
                        {"url": "https://cnn.com/b"},
                        {"url": "https://reuters.com/c"},
                    ],
                },
            ],
        })
        pfr = result.per_fact_results[0]
        assert pfr.unique_sources == 3
        assert pfr.unique_domains == 2

    def test_per_fact_confidence(self):
        """Per-fact results contain correct confidence."""
        result = evaluate_evidence({
            "facts": [
                {"fact": "A", "confidence": 0.75, "evidence": []},
            ],
        })
        assert result.per_fact_results[0].confidence == 0.75


# ==========================================
# DATA MODEL TESTS
# ==========================================


class TestDataModels:
    """Tests for dataclass models."""

    def test_evidence_evaluation_result_defaults(self):
        """Default values for EvidenceEvaluationResult."""
        r = EvidenceEvaluationResult()
        assert r.total_facts == 0
        assert r.support_ratio == 0.0
        assert r.citation_count == 0
        assert r.average_fact_confidence == 0.0
        assert r.coverage_score == 0.0
        assert r.per_fact_results == []

    def test_evidence_evaluation_result_frozen(self):
        """EvidenceEvaluationResult is immutable."""
        r = EvidenceEvaluationResult(total_facts=5)
        with pytest.raises(AttributeError):
            r.total_facts = 10  # type: ignore[misc]

    def test_evidence_fact_result_defaults(self):
        """Default values for EvidenceFactResult."""
        r = EvidenceFactResult()
        assert r.fact_text == ""
        assert r.confidence == 0.0
        assert r.evidence_count == 0
        assert r.supported is False
        assert r.unique_sources == 0
        assert r.unique_domains == 0

    def test_evidence_fact_result_frozen(self):
        """EvidenceFactResult is immutable."""
        r = EvidenceFactResult(fact_text="test")
        with pytest.raises(AttributeError):
            r.fact_text = "changed"  # type: ignore[misc]


# ==========================================
# IMPORT TESTS
# ==========================================


class TestPackageImports:
    """Tests that package exports work correctly."""

    def test_from_evaluation_import(self):
        """Import from app.evaluation includes evidence evaluator."""
        from app.evaluation import (
            evaluate_evidence,
            EvidenceEvaluationResult,
            EvidenceFactResult,
        )
        assert callable(evaluate_evidence)
        assert EvidenceEvaluationResult is not None
        assert EvidenceFactResult is not None

    def test_direct_import(self):
        """Direct import from evidence_evaluator module works."""
        from app.evaluation.evidence_evaluator import (
            evaluate_evidence,
            EvidenceEvaluationResult,
            EvidenceFactResult,
        )
        assert callable(evaluate_evidence)
        assert EvidenceEvaluationResult is not None
        assert EvidenceFactResult is not None

    def test_no_existing_test_breakage(self):
        """Confirm existing retrieval evaluation tests still import correctly."""
        from app.evaluation import (
            RetrievalEvaluator,
            RetrievalMetrics,
            ChunkScore,
            compute_jaccard_similarity,
        )
        assert RetrievalEvaluator is not None
        assert RetrievalMetrics is not None
        assert ChunkScore is not None
        assert callable(compute_jaccard_similarity)
