# ==========================================
# HIERARCHICAL REPORT GENERATION TESTS
# ==========================================
#
# Tests for the refactored report_agent.py
# covering section generation, assembly,
# graceful degradation, and metrics.
#
# ==========================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from app.agents.report_agent import (
    SECTION_CONCURRENCY,
    _build_citations,
    _build_fact_sections,
    assemble_report,
    report_agent,
)


# ==========================================
# FIXTURES
# ==========================================


@pytest.fixture
def sample_summaries():
    return [
        {
            "source": "Article A",
            "url": "https://example.com/a",
            "facts": [
                {
                    "fact": "Fact one from A",
                    "source_url": "https://example.com/a",
                    "source_title": "Article A",
                },
                {
                    "fact": "Fact two from A",
                    "source_url": "https://example.com/a",
                    "source_title": "Article A",
                },
            ],
        },
        {
            "source": "Article B",
            "url": "https://example.com/b",
            "facts": [
                {
                    "fact": "Fact one from B",
                    "source_url": "https://example.com/b",
                    "source_title": "Article B",
                }
            ],
        },
    ]


@pytest.fixture
def sample_verified_facts():
    return {
        "confirmed_facts": [
            {"fact": "Confirmed fact 1", "confidence": 0.91},
            {"fact": "Confirmed fact 2", "confidence": 0.85},
        ],
        "disputed_facts": [
            {"fact": "Disputed fact 1", "confidence": 0.45},
        ],
        "low_confidence_facts": [
            {"fact": "Low confidence fact 1", "confidence": 0.32},
        ],
    }


# ==========================================
# PROMPT BUILDER TESTS
# ==========================================


class TestBuildFactSections:
    def test_confirmed_only(self):
        facts = {"confirmed_facts": [{"fact": "A", "confidence": 0.9}]}
        confirmed, disputed, low = _build_fact_sections(facts)
        assert "- A (confidence: 0.9)" in confirmed
        assert disputed == ""
        assert low == ""

    def test_all_categories(self, sample_verified_facts):
        confirmed, disputed, low = _build_fact_sections(sample_verified_facts)
        assert "Confirmed fact 1" in confirmed
        assert "Confirmed fact 2" in confirmed
        assert "Disputed fact 1" in disputed
        assert "Low confidence fact 1" in low

    def test_empty(self):
        confirmed, disputed, low = _build_fact_sections({})
        assert confirmed == ""
        assert disputed == ""
        assert low == ""

    def test_string_facts_fallback(self):
        facts = {"confirmed_facts": ["Raw string fact"]}
        confirmed, disputed, low = _build_fact_sections(facts)
        assert "- Raw string fact" in confirmed

    def test_dict_fact_without_confidence(self):
        facts = {"confirmed_facts": [{"fact": "No confidence fact"}]}
        confirmed, _, _ = _build_fact_sections(facts)
        assert "- No confidence fact" in confirmed


class TestBuildCitations:
    def test_from_summaries(self, sample_summaries):
        sources, refs, entries = _build_citations(sample_summaries, {})
        assert "https://example.com/a" in sources
        assert "https://example.com/b" in sources
        assert "[1]" in sources
        assert "[2]" in sources
        assert refs != ""

    def test_from_factcheck_evidence(self, sample_summaries):
        verified = {
            "confirmed_facts": [
                {
                    "fact": "Test",
                    "evidence": [{"url": "https://example.com/c"}],
                }
            ]
        }
        sources, refs, entries = _build_citations(sample_summaries, verified)
        assert "https://example.com/c" in sources
        assert len(entries) == 3  # a, b, c

    def test_no_duplicates(self, sample_summaries):
        """Same URL from multiple sources should only appear once."""
        verified = {
            "confirmed_facts": [
                {
                    "fact": "Test",
                    "evidence": [{"url": "https://example.com/a"}],
                }
            ]
        }
        sources, refs, entries = _build_citations(sample_summaries, verified)
        # Count how many times https://example.com/a appears in sources
        count = sources.count("https://example.com/a")
        assert count == 1

    def test_empty(self):
        sources, refs, entries = _build_citations([], {})
        assert sources == "(no sources available)"
        assert refs == ""
        assert entries == []

    def test_no_urls_in_summaries(self):
        summaries = [{"source": "No URL", "facts": []}]
        sources, refs, entries = _build_citations(summaries, {})
        assert sources == "(no sources available)"


# ==========================================
# ASSEMBLY TESTS
# ==========================================


class TestAssembleReport:
    def test_all_sections_succeed(self):
        sections = {
            "executive_summary": "Exec summary text",
            "key_findings": "Key findings text",
            "evidence_analysis": "Evidence analysis text",
            "limitations": "Limitations text",
        }
        status = {k: "success" for k in sections}
        report = assemble_report(sections, status, "[1] Source: url")
        assert "Exec summary text" in report
        assert "Key findings text" in report
        assert "Evidence analysis text" in report
        assert "Limitations text" in report
        assert "[1] Source: url" in report
        assert "## Sources" in report

    def test_sections_in_correct_order(self):
        sections = {
            "executive_summary": "AAA",
            "key_findings": "BBB",
            "evidence_analysis": "CCC",
            "limitations": "DDD",
        }
        status = {k: "success" for k in sections}
        report = assemble_report(sections, status, "[1] Source")
        aaa_pos = report.index("AAA")
        bbb_pos = report.index("BBB")
        ccc_pos = report.index("CCC")
        ddd_pos = report.index("DDD")
        sources_pos = report.index("## Sources")
        assert aaa_pos < bbb_pos < ccc_pos < ddd_pos < sources_pos

    def test_one_section_fails(self):
        sections = {
            "executive_summary": "Exec summary text",
            "key_findings": None,
            "evidence_analysis": "Evidence analysis text",
            "limitations": "Limitations text",
        }
        status = {
            "executive_summary": "success",
            "key_findings": "failed",
            "evidence_analysis": "success",
            "limitations": "success",
        }
        report = assemble_report(sections, status, "[1] Source")
        assert "Exec summary text" in report
        assert "Key findings text" not in report
        assert "Evidence analysis text" in report
        assert "Limitations text" in report

    def test_multiple_sections_fail(self):
        sections = {
            "executive_summary": None,
            "key_findings": "Key findings text",
            "evidence_analysis": None,
            "limitations": "Limitations text",
        }
        status = {
            "executive_summary": "failed",
            "key_findings": "success",
            "evidence_analysis": "failed",
            "limitations": "success",
        }
        report = assemble_report(sections, status, "[1] Source")
        assert "Key findings text" in report
        assert "Limitations text" in report
        assert "Exec summary text" not in report
        assert "Evidence analysis text" not in report

    def test_all_sections_fail(self):
        sections = {k: None for k in ("executive_summary", "key_findings", "evidence_analysis", "limitations")}
        status = {k: "failed" for k in sections}
        report = assemble_report(sections, status, "[1] Source")
        assert "Report generation failed for all sections" in report
        assert "[1] Source" in report

    def test_backward_compatible_format(self):
        """The assembled report should have the same structure as before."""
        sections = {
            "executive_summary": "Executive Summary\n\nFirst paragraph.",
            "key_findings": "Key Findings\n\n1. Finding one [1].",
            "evidence_analysis": "Evidence Analysis\n\nStrong evidence.",
            "limitations": "Limitations\n\nSome uncertainty.",
        }
        status = {k: "success" for k in sections}
        report = assemble_report(sections, status, "[1] Example: url")

        # The report must:
        # - Be a string
        assert isinstance(report, str)
        # - Contain all sections
        assert "Executive Summary" in report
        assert "Key Findings" in report
        assert "Evidence Analysis" in report
        assert "Limitations" in report
        # - Have --- separator
        assert "---" in report
        # - Have Sources heading
        assert "## Sources" in report
        # - End with the sources section
        assert report.strip().endswith("[1] Example: url")


# ==========================================
# REPORT AGENT METRICS TESTS
# ==========================================


class TestReportAgentMetrics:
    @patch("app.agents.report_agent._generate_executive_summary")
    @patch("app.agents.report_agent._generate_key_findings")
    @patch("app.agents.report_agent._generate_evidence_analysis")
    @patch("app.agents.report_agent._generate_limitations")
    @pytest.mark.asyncio
    async def test_metrics_structure(
        self,
        mock_limitations,
        mock_evidence,
        mock_key_findings,
        mock_exec,
    ):
        mock_exec.return_value = ("Exec summary", 100.0)
        mock_key_findings.return_value = ("Key findings", 200.0)
        mock_evidence.return_value = ("Evidence analysis", 150.0)
        mock_limitations.return_value = ("Limitations text", 120.0)

        report, metrics = await report_agent(
            "Test topic",
            [],
            {"confirmed_facts": [], "disputed_facts": [], "low_confidence_facts": []},
        )

        # Check metrics keys
        assert "executive_summary_time_ms" in metrics
        assert "key_findings_time_ms" in metrics
        assert "evidence_analysis_time_ms" in metrics
        assert "limitations_time_ms" in metrics
        assert "total_report_time_ms" in metrics
        assert "section_failures" in metrics
        assert "successful_sections" in metrics
        assert "failed_sections" in metrics
        assert "citations_generated" in metrics

        # All sections succeeded
        assert metrics["section_failures"] == 0
        assert metrics["successful_sections"] == 4
        assert metrics["failed_sections"] == 0

    @patch("app.agents.report_agent._generate_executive_summary")
    @patch("app.agents.report_agent._generate_key_findings")
    @patch("app.agents.report_agent._generate_evidence_analysis")
    @patch("app.agents.report_agent._generate_limitations")
    @pytest.mark.asyncio
    async def test_section_failure_tracking(
        self,
        mock_limitations,
        mock_evidence,
        mock_key_findings,
        mock_exec,
    ):
        """Two sections fail, metrics should reflect that."""
        mock_exec.return_value = ("Exec summary", 100.0)
        mock_key_findings.return_value = (None, 200.0)  # failed
        mock_evidence.return_value = ("Evidence analysis", 150.0)
        mock_limitations.return_value = (None, 120.0)  # failed

        report, metrics = await report_agent(
            "Test topic",
            [],
            {"confirmed_facts": [], "disputed_facts": [], "low_confidence_facts": []},
        )

        assert metrics["section_failures"] == 2
        assert metrics["successful_sections"] == 2
        assert metrics["failed_sections"] == 2

    @patch("app.agents.report_agent._generate_executive_summary")
    @patch("app.agents.report_agent._generate_key_findings")
    @patch("app.agents.report_agent._generate_evidence_analysis")
    @patch("app.agents.report_agent._generate_limitations")
    @pytest.mark.asyncio
    async def test_report_is_string(
        self,
        mock_limitations,
        mock_evidence,
        mock_key_findings,
        mock_exec,
    ):
        mock_exec.return_value = ("Exec", 100.0)
        mock_key_findings.return_value = ("Findings", 200.0)
        mock_evidence.return_value = ("Evidence", 150.0)
        mock_limitations.return_value = ("Limits", 120.0)

        report, metrics = await report_agent("Test topic", [], {"confirmed_facts": [], "disputed_facts": [], "low_confidence_facts": []})

        assert isinstance(report, str)
        assert len(report) > 0

    @patch("app.agents.report_agent._generate_executive_summary")
    @patch("app.agents.report_agent._generate_key_findings")
    @patch("app.agents.report_agent._generate_evidence_analysis")
    @patch("app.agents.report_agent._generate_limitations")
    @pytest.mark.asyncio
    async def test_partial_report_on_failure(
        self,
        mock_limitations,
        mock_evidence,
        mock_key_findings,
        mock_exec,
    ):
        """Partial report should only contain successful sections."""
        mock_exec.return_value = ("Exec summary here", 100.0)
        mock_key_findings.return_value = (None, 200.0)  # failed
        mock_evidence.return_value = ("Evidence analysis here", 150.0)
        mock_limitations.return_value = (None, 120.0)  # failed

        report, metrics = await report_agent("Test topic", [], {"confirmed_facts": [], "disputed_facts": [], "low_confidence_facts": []})

        assert "Exec summary here" in report
        assert "Evidence analysis here" in report
        # These should not appear since the sections failed
        assert "Report generation failed for all sections" not in report

    @patch("app.agents.report_agent._generate_executive_summary")
    @patch("app.agents.report_agent._generate_key_findings")
    @patch("app.agents.report_agent._generate_evidence_analysis")
    @patch("app.agents.report_agent._generate_limitations")
    @pytest.mark.asyncio
    async def test_citations_count(
        self,
        mock_limitations,
        mock_evidence,
        mock_key_findings,
        mock_exec,
    ):
        mock_exec.return_value = ("Exec", 100.0)
        mock_key_findings.return_value = ("Findings", 200.0)
        mock_evidence.return_value = ("Evidence", 150.0)
        mock_limitations.return_value = ("Limits", 120.0)

        summaries = [
            {
                "source": "Article A",
                "url": "https://example.com/a",
                "facts": [],
            },
        ]

        report, metrics = await report_agent("Test topic", summaries, {"confirmed_facts": [], "disputed_facts": [], "low_confidence_facts": []})

        # One source = one citation
        assert metrics["citations_generated"] == 1


# ==========================================
# CONFIGURATION TESTS
# ==========================================


class TestConfiguration:
    def test_section_concurrency_default(self):
        assert SECTION_CONCURRENCY == 1


# ==========================================
# RETRY PRESERVATION TESTS
# ==========================================


class TestSectionGeneratorErrorHandling:
    """Verify that section generators preserve retry/timeout patterns."""

    @pytest.mark.asyncio
    async def test_all_generators_use_run_agent(self):
        """All section generators call run_agent under the hood."""
        from app.agents.report_agent import (
            _generate_executive_summary,
            _generate_key_findings,
            _generate_evidence_analysis,
            _generate_limitations,
        )

        with patch("app.agents.report_agent.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Generated content"

            # Test each generator
            content, timing = await _generate_executive_summary("topic", "confirmed", "disputed", "low", "[1] ref")
            assert content == "Generated content"
            assert timing > 0
            assert mock_run.call_count >= 1

            mock_run.reset_mock()
            content, timing = await _generate_key_findings("topic", "confirmed", "[1] ref")
            assert content == "Generated content"
            assert mock_run.call_count >= 1

            mock_run.reset_mock()
            content, timing = await _generate_evidence_analysis("topic", "confirmed", "[1] ref")
            assert content == "Generated content"
            assert mock_run.call_count >= 1

            mock_run.reset_mock()
            content, timing = await _generate_limitations("topic", "disputed", "low", "[1] ref")
            assert content == "Generated content"
            assert mock_run.call_count >= 1

    @pytest.mark.asyncio
    async def test_generator_failure_returns_none(self):
        """When run_agent raises, generator should return (None, timing)."""
        with patch("app.agents.report_agent.run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("LLM error")

            from app.agents.report_agent import _generate_executive_summary

            content, timing = await _generate_executive_summary("topic", "confirmed", "disputed", "low", "[1] ref")
            assert content is None
            assert timing > 0
