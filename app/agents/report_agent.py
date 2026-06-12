# ==========================================
# HIERARCHICAL REPORT GENERATION v1
# ==========================================
#
# Replaces the monolithic prompt approach with
# independent section-based generation.
#
# Sections:
#   1. Executive Summary
#   2. Key Findings
#   3. Evidence Analysis
#   4. Limitations and Uncertainty
#   5. Sources (code-generated, not LLM)
#
# Each section is generated independently with
# its own LLM call. If one section fails, the
# remaining sections continue. A final assembly
# step combines all sections into the exact
# same report format as the previous monolithic
# approach — ensuring backward compatibility.
#
# Usage (backward-compatible):
#     report, metrics = await report_agent(topic, summaries, verified_facts)
#
# ==========================================

import asyncio
import time

from typing import Dict, List, Optional, Tuple

from app.services.llm_service import run_agent

from app.config.settings import settings

from app.utils.logger import logger


# ==========================================
# CONCURRENCY CONTROL
# ==========================================

SECTION_CONCURRENCY = 1
_section_semaphore = asyncio.Semaphore(SECTION_CONCURRENCY)


# ==========================================
# PROMPT BUILDERS
# ==========================================


def _build_fact_sections(verified_facts: dict) -> Tuple[str, str, str]:
    """Extract and format confirmed, disputed, and low-confidence facts."""

    def _format_fact(f):
        if isinstance(f, dict):
            fact_text = f.get("fact", str(f))
            confidence = f.get("confidence", None)
            if confidence is not None:
                return f"- {fact_text} (confidence: {confidence})"
            return f"- {fact_text}"
        return f"- {f}"

    confirmed = "\n".join(
        _format_fact(f)
        for f in verified_facts.get("confirmed_facts", [])
    )

    disputed = "\n".join(
        _format_fact(f)
        for f in verified_facts.get("disputed_facts", [])
    )

    low_confidence = "\n".join(
        _format_fact(f)
        for f in verified_facts.get("low_confidence_facts", [])
    )

    return confirmed, disputed, low_confidence


def _build_citations(
    summaries: List[dict],
    verified_facts: dict,
) -> Tuple[str, str, List[Tuple[str, str]]]:
    """Build deterministic citations (code-generated, not LLM)."""
    seen_urls = set()
    citation_entries: List[Tuple[str, str]] = []

    for s in summaries:
        for fact_entry in s.get("facts", []):
            if isinstance(fact_entry, dict):
                src_url = fact_entry.get("source_url", "")
                if src_url and src_url not in seen_urls:
                    seen_urls.add(src_url)
                    citation_entries.append((
                        fact_entry.get("source_title", "Source"),
                        src_url,
                    ))

        url = s.get("url", "")
        title = s.get("source", "Source")
        if url and url not in seen_urls:
            seen_urls.add(url)
            citation_entries.append((title, url))

    for category_key in ("confirmed_facts", "disputed_facts", "low_confidence_facts"):
        for f in verified_facts.get(category_key, []):
            if isinstance(f, dict):
                for ev in f.get("evidence", []):
                    ev_url = ev.get("url", "")
                    if ev_url and ev_url not in seen_urls:
                        seen_urls.add(ev_url)
                        citation_entries.append(("Source", ev_url))

    source_lines = []
    for idx, (title, url) in enumerate(citation_entries, 1):
        source_lines.append(f"[{idx}] {title}: {url}")

    sources_section = "\n".join(source_lines) if source_lines else "(no sources available)"

    citation_refs = ", ".join(
        f"[{n}]" for n in range(1, len(citation_entries) + 1)
    ) if citation_entries else ""

    return sources_section, citation_refs, citation_entries


# ==========================================
# SECTION GENERATORS
# ==========================================


async def _generate_executive_summary(
    topic: str,
    confirmed: str,
    disputed: str,
    low_confidence: str,
    citation_refs: str,
) -> Tuple[Optional[str], float]:
    start = time.time()
    logger.info("[REPORT SECTION] Executive Summary started")

    try:
        prompt = f"""\
Write an Executive Summary for a research report on:

{topic}

CONFIRMED FACTS:
{confirmed}

DISPUTED FACTS:
{disputed if disputed else "None"}

LOW CONFIDENCE FACTS:
{low_confidence}

CITATIONS: Reference sources using bracketed numbers: {citation_refs}

Provide a concise 2-3 paragraph executive summary covering the most important findings.
Cite sources using [1], [2], etc.
"""

        content = await run_agent(
            "You are a professional research analyst writing an executive summary.",
            prompt,
            max_tokens=settings.REPORT_MAX_TOKENS // 3,
        )

        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.info("[REPORT SECTION] Executive Summary completed in %.2fms", elapsed_ms)
        return content, elapsed_ms

    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error("[REPORT SECTION] Executive Summary failed after %.2fms: %s", elapsed_ms, e)
        return None, elapsed_ms


async def _generate_key_findings(
    topic: str,
    confirmed: str,
    citation_refs: str,
) -> Tuple[Optional[str], float]:
    start = time.time()
    logger.info("[REPORT SECTION] Key Findings started")

    try:
        prompt = f"""\
Write the Key Findings section for a research report on:

{topic}

CONFIRMED FACTS:
{confirmed}

CITATIONS: Reference sources using bracketed numbers: {citation_refs}

List the key findings in order of importance. Each finding should have:
- A clear statement of the finding
- Citation references using [1], [2], etc.
- A brief explanation

Be factual and concise.
"""

        content = await run_agent(
            "You are a professional research analyst writing key findings.",
            prompt,
            max_tokens=settings.REPORT_MAX_TOKENS // 3,
        )

        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.info("[REPORT SECTION] Key Findings completed in %.2fms", elapsed_ms)
        return content, elapsed_ms

    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error("[REPORT SECTION] Key Findings failed after %.2fms: %s", elapsed_ms, e)
        return None, elapsed_ms


async def _generate_evidence_analysis(
    topic: str,
    confirmed: str,
    citation_refs: str,
) -> Tuple[Optional[str], float]:
    start = time.time()
    logger.info("[REPORT SECTION] Evidence Analysis started")

    try:
        prompt = f"""\
Write the Evidence Analysis section for a research report on:

{topic}

CONFIRMED FACTS:
{confirmed}

CITATIONS: Reference sources using bracketed numbers: {citation_refs}

Analyze the quality and strength of the evidence supporting the findings.
Discuss:
1. Which findings have the strongest evidence support
2. Any patterns in the evidence
3. The overall reliability of the evidence base

Cite sources using [1], [2], etc.
"""

        content = await run_agent(
            "You are a professional research analyst writing an evidence analysis.",
            prompt,
            max_tokens=settings.REPORT_MAX_TOKENS // 3,
        )

        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.info("[REPORT SECTION] Evidence Analysis completed in %.2fms", elapsed_ms)
        return content, elapsed_ms

    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error("[REPORT SECTION] Evidence Analysis failed after %.2fms: %s", elapsed_ms, e)
        return None, elapsed_ms


async def _generate_limitations(
    topic: str,
    disputed: str,
    low_confidence: str,
    citation_refs: str,
) -> Tuple[Optional[str], float]:
    start = time.time()
    logger.info("[REPORT SECTION] Limitations and Uncertainty started")

    try:
        prompt = f"""\
Write the Limitations and Uncertainty section for a research report on:

{topic}

DISPUTED FACTS:
{disputed if disputed else "None"}

LOW CONFIDENCE FACTS:
{low_confidence if low_confidence else "None"}

CITATIONS: Reference sources using bracketed numbers: {citation_refs}

Discuss:
1. Areas where the evidence is conflicting or disputed
2. Claims with low confidence that need further investigation
3. Gaps in the current understanding
4. Recommendations for further research

Cite sources using [1], [2], etc.
"""

        content = await run_agent(
            "You are a professional research analyst writing about limitations and uncertainty.",
            prompt,
            max_tokens=settings.REPORT_MAX_TOKENS // 3,
        )

        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.info("[REPORT SECTION] Limitations completed in %.2fms", elapsed_ms)
        return content, elapsed_ms

    except Exception as e:
        elapsed_ms = round((time.time() - start) * 1000, 2)
        logger.error("[REPORT SECTION] Limitations failed after %.2fms: %s", elapsed_ms, e)
        return None, elapsed_ms


# ==========================================
# ASSEMBLY
# ==========================================


def assemble_report(
    sections: Dict[str, Optional[str]],
    section_status: Dict[str, str],
    sources_section: str,
) -> str:
    """Assemble the final report from individual sections.

    Args:
        sections: Dict of section_name -> content (None if failed).
        section_status: Dict of section_name -> "success" or "failed".
        sources_section: The code-generated sources section text.

    Returns:
        A complete report string in the same format as the
        previous monolithic generation.
    """
    parts: List[str] = []

    if sections.get("executive_summary"):
        parts.append(sections["executive_summary"].strip())

    if sections.get("key_findings"):
        parts.append(sections["key_findings"].strip())

    if sections.get("evidence_analysis"):
        parts.append(sections["evidence_analysis"].strip())

    if sections.get("limitations"):
        parts.append(sections["limitations"].strip())

    body = "\n\n".join(parts) if parts else "(Report generation failed for all sections)"

    report = body + "\n\n---\n\n## Sources\n\n" + sources_section

    return report


# ==========================================
# MAIN REPORT AGENT
# ==========================================


async def report_agent(
    topic: str,
    summaries: List[dict],
    verified_facts: dict,
) -> Tuple[str, dict]:
    """Generate a complete research report using hierarchical section generation.

    This function replaces the previous monolithic prompt approach.
    The return value is backward-compatible: the first element is the
    report string, identical in format to the previous version.

    Args:
        topic: The research topic.
        summaries: List of summary dicts from the summarizer agent.
        verified_facts: Dict with confirmed_facts, disputed_facts,
                       low_confidence_facts from the factcheck agent.

    Returns:
        A tuple of (report_string, metrics_dict) with per-section timing metrics.
    """
    logger.info("[REPORT AGENT] Starting hierarchical report generation")
    total_start = time.time()

    confirmed, disputed, low_confidence = _build_fact_sections(verified_facts)
    sources_section, citation_refs, citation_entries = _build_citations(summaries, verified_facts)

    sections: Dict[str, Optional[str]] = {}
    section_timing: Dict[str, float] = {}
    section_status: Dict[str, str] = {}

    generators = {
        "executive_summary": _generate_executive_summary(
            topic, confirmed, disputed, low_confidence, citation_refs,
        ),
        "key_findings": _generate_key_findings(
            topic, confirmed, citation_refs,
        ),
        "evidence_analysis": _generate_evidence_analysis(
            topic, confirmed, citation_refs,
        ),
        "limitations": _generate_limitations(
            topic, disputed, low_confidence, citation_refs,
        ),
    }

    async def _run_section(name: str, coro) -> None:
        async with _section_semaphore:
            content, timing_ms = await coro
            sections[name] = content
            section_timing[name] = timing_ms
            section_status[name] = "success" if content is not None else "failed"

    tasks = [
        _run_section(name, coro)
        for name, coro in generators.items()
    ]
    await asyncio.gather(*tasks)

    report = assemble_report(sections, section_status, sources_section)

    total_report_time = round((time.time() - total_start) * 1000, 2)
    successful = sum(1 for s in section_status.values() if s == "success")
    failed = sum(1 for s in section_status.values() if s == "failed")

    # NEW METRICS: per-section timing and failure tracking
    report_metrics = {
        "executive_summary_time_ms": section_timing.get("executive_summary", 0.0),
        "key_findings_time_ms": section_timing.get("key_findings", 0.0),
        "evidence_analysis_time_ms": section_timing.get("evidence_analysis", 0.0),
        "limitations_time_ms": section_timing.get("limitations", 0.0),
        "total_report_time_ms": total_report_time,
        "section_failures": failed,
        "successful_sections": successful,
        "failed_sections": failed,
        "citations_generated": len(citation_entries),
    }

    logger.info(
        "[REPORT AGENT] Hierarchical generation complete | "
        "total=%.2fms | sections=%d/%d | "
        "exec_summary=%.2fms | key_findings=%.2fms | "
        "evidence=%.2fms | limitations=%.2fms",
        total_report_time,
        successful,
        successful + failed,
        section_timing.get("executive_summary", 0.0),
        section_timing.get("key_findings", 0.0),
        section_timing.get("evidence_analysis", 0.0),
        section_timing.get("limitations", 0.0),
    )

    if failed > 0:
        logger.warning(
            "[REPORT AGENT] %d section(s) failed: %s",
            failed,
            [name for name, status in section_status.items() if status == "failed"],
        )

    return report, report_metrics
