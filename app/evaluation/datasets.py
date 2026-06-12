# ==========================================
# RETRIEVAL EVALUATION — BENCHMARK DATASETS
# ==========================================
#
# A curated set of benchmark queries for
# measuring retrieval quality across diverse
# knowledge domains.
#
# Usage:
#     from app.evaluation.datasets import (
#         DEFAULT_BENCHMARK_QUERIES,
#         BenchmarkQuery,
#     )
#     for entry in DEFAULT_BENCHMARK_QUERIES:
#         print(entry.query)
#
# ==========================================

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BenchmarkQuery:
    """A single benchmark query with metadata.

    Attributes:
        id:         Unique identifier for the query (e.g. ``"hist-001"``).
        query:      The search query text to run against the retrieval system.
        category:   Knowledge domain category (e.g. ``"history"``, ``"science"``).
        notes:      Optional context about what the query targets.
    """

    id: str
    query: str
    category: str
    notes: str = ""


# ==========================================
# DEFAULT BENCHMARK QUERIES
# ==========================================
#
# 35 queries covering 7 categories:
#   1. History          (hist-001 – hist-006)
#   2. Science           (sci-007  – sci-012)
#   3. Technology        (tech-013 – tech-018)
#   4. Medicine          (med-019  – med-024)
#   5. Current Affairs   (curr-025 – curr-030)
#   6. General Knowledge (gen-031  – gen-035)
#
# ==========================================

DEFAULT_BENCHMARK_QUERIES: List[BenchmarkQuery] = [
    # ── History ─────────────────────────────────────────────
    BenchmarkQuery(
        id="hist-001",
        query="Causes and key events of World War II",
        category="history",
        notes="Broad historical overview query",
    ),
    BenchmarkQuery(
        id="hist-002",
        query="The fall of the Roman Empire and its lasting impact on Europe",
        category="history",
        notes="Causal & impact analysis query",
    ),
    BenchmarkQuery(
        id="hist-003",
        query="Major figures and outcomes of the American Civil Rights Movement",
        category="history",
        notes="Named-entity & event-based query",
    ),
    BenchmarkQuery(
        id="hist-004",
        query="Economic consequences of the Industrial Revolution",
        category="history",
        notes="Economic history query",
    ),
    BenchmarkQuery(
        id="hist-005",
        query="Causes and legacy of the French Revolution",
        category="history",
        notes="Revolutionary history query",
    ),
    BenchmarkQuery(
        id="hist-006",
        query="Origins and impact of the Cold War on global politics",
        category="history",
        notes="20th century geopolitical query",
    ),
    # ── Science ─────────────────────────────────────────────
    BenchmarkQuery(
        id="sci-007",
        query="How does natural selection drive evolutionary change in species",
        category="science",
        notes="Mechanism-explanation query",
    ),
    BenchmarkQuery(
        id="sci-008",
        query="Quantum entanglement and its implications for secure communication",
        category="science",
        notes="Advanced physics concept query",
    ),
    BenchmarkQuery(
        id="sci-009",
        query="The role of gut microbiota in human health and disease",
        category="science",
        notes="Biology / health science query",
    ),
    BenchmarkQuery(
        id="sci-010",
        query="Plate tectonics and the formation of mountain ranges",
        category="science",
        notes="Earth science query",
    ),
    BenchmarkQuery(
        id="sci-011",
        query="Dark matter and dark energy in cosmological models",
        category="science",
        notes="Astrophysics / cosmology query",
    ),
    BenchmarkQuery(
        id="sci-012",
        query="CRISPR gene editing technology and its ethical implications",
        category="science",
        notes="Biotechnology ethics query",
    ),
    # ── Technology ──────────────────────────────────────────
    BenchmarkQuery(
        id="tech-013",
        query="Differences between supervised and unsupervised machine learning",
        category="technology",
        notes="Concept-comparison query",
    ),
    BenchmarkQuery(
        id="tech-014",
        query="How blockchain technology works beyond cryptocurrency",
        category="technology",
        notes="Explanation-of-technology query",
    ),
    BenchmarkQuery(
        id="tech-015",
        query="Architecture and performance of transformer neural networks",
        category="technology",
        notes="Deep learning / NLP query",
    ),
    BenchmarkQuery(
        id="tech-016",
        query="Edge computing versus cloud computing for IoT applications",
        category="technology",
        notes="Technology comparison query",
    ),
    BenchmarkQuery(
        id="tech-017",
        query="Zero trust architecture in cybersecurity",
        category="technology",
        notes="Cybersecurity architecture query",
    ),
    BenchmarkQuery(
        id="tech-018",
        query="Serverless computing versus container orchestration",
        category="technology",
        notes="Cloud infrastructure comparison",
    ),
    # ── Medicine ────────────────────────────────────────────
    BenchmarkQuery(
        id="med-019",
        query="Mechanism of action of mRNA vaccines",
        category="medicine",
        notes="Biomedical mechanism query",
    ),
    BenchmarkQuery(
        id="med-020",
        query="Risk factors and early detection methods for Alzheimer's disease",
        category="medicine",
        notes="Disease risk & diagnosis query",
    ),
    BenchmarkQuery(
        id="med-021",
        query="Effectiveness of cognitive behavioral therapy for anxiety disorders",
        category="medicine",
        notes="Therapeutic intervention query",
    ),
    BenchmarkQuery(
        id="med-022",
        query="Global antibiotic resistance trends and mitigation strategies",
        category="medicine",
        notes="Public health query",
    ),
    BenchmarkQuery(
        id="med-023",
        query="Role of epigenetics in cancer development",
        category="medicine",
        notes="Molecular oncology query",
    ),
    BenchmarkQuery(
        id="med-024",
        query="Comparison of diagnostic imaging techniques: MRI versus CT scans",
        category="medicine",
        notes="Medical imaging comparison",
    ),
    # ── Current Affairs ─────────────────────────────────────
    BenchmarkQuery(
        id="curr-025",
        query="International climate agreements and their emissions reduction targets",
        category="current_affairs",
        notes="Environmental policy query",
    ),
    BenchmarkQuery(
        id="curr-026",
        query="Economic impact of remote work on urban real estate markets",
        category="current_affairs",
        notes="Post-pandemic economic query",
    ),
    BenchmarkQuery(
        id="curr-027",
        query="Artificial intelligence regulation frameworks in the European Union",
        category="current_affairs",
        notes="AI governance policy query",
    ),
    BenchmarkQuery(
        id="curr-028",
        query="Global supply chain disruptions and reshoring trends",
        category="current_affairs",
        notes="Trade and logistics query",
    ),
    BenchmarkQuery(
        id="curr-029",
        query="Cybersecurity threats from state-sponsored hacking groups",
        category="current_affairs",
        notes="Geopolitical cybersecurity query",
    ),
    BenchmarkQuery(
        id="curr-030",
        query="Impact of central bank digital currencies on traditional banking",
        category="current_affairs",
        notes="Financial technology policy query",
    ),
    # ── General Knowledge ───────────────────────────────────
    BenchmarkQuery(
        id="gen-031",
        query="Fundamental principles of macroeconomic theory",
        category="general_knowledge",
        notes="Economics fundamentals query",
    ),
    BenchmarkQuery(
        id="gen-032",
        query="Key differences between civil law and common law legal systems",
        category="general_knowledge",
        notes="Legal systems comparison query",
    ),
    BenchmarkQuery(
        id="gen-033",
        query="Major schools of philosophy and their core ideas",
        category="general_knowledge",
        notes="Philosophy survey query",
    ),
    BenchmarkQuery(
        id="gen-034",
        query="Principles of renewable energy systems and grid integration",
        category="general_knowledge",
        notes="Energy systems query",
    ),
    BenchmarkQuery(
        id="gen-035",
        query="Evolution of human language and linguistic theory",
        category="general_knowledge",
        notes="Linguistics query",
    ),
]


# ==========================================
# BENCHMARK FAILURE QUERIES
# ==========================================
#
# Queries designed to stress-test the
# retrieval system's edge-case handling.
# These are NOT included in
# DEFAULT_BENCHMARK_QUERIES by default;
# they can be appended or run separately.
#
# ==========================================

BENCHMARK_FAILURE_QUERIES: List[BenchmarkQuery] = [
    BenchmarkQuery(
        id="fail-001",
        query="",
        category="failure",
        notes="Empty query — should be rejected",
    ),
    BenchmarkQuery(
        id="fail-002",
        query="      ",
        category="failure",
        notes="Whitespace-only query — should be rejected",
    ),
    BenchmarkQuery(
        id="fail-003",
        query="asdfghjkl",
        category="failure",
        notes="Gibberish — tests low-relevance retrieval",
    ),
    BenchmarkQuery(
        id="fail-004",
        query="Tell me everything",
        category="failure",
        notes="Overly broad query — tests retrieval limits",
    ),
    BenchmarkQuery(
        id="fail-005",
        query="COVID and quantum mechanics",
        category="failure",
        notes="Cross-domain nonsense — tests handling of unrelated concepts",
    ),
]
