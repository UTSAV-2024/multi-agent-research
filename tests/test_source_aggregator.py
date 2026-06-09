# ==========================================
# SOURCE AGGREGATOR TESTS
# ==========================================
#
# Tests for the source aggregator service.
#
# ==========================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.source_aggregator import (
    aggregate_sources,
    build_fact_extraction_context,
    estimate_context_size,
)


class TestAggregateSources:
    """Tests for aggregate_sources()."""

    def test_empty_sources_returns_empty_string(self):
        """aggregate_sources([]) returns empty string."""
        result = aggregate_sources([])
        assert result == ""

    def test_single_source(self):
        """aggregate_sources() formats a single source correctly."""
        sources = [
            {
                "title": "Test Article",
                "url": "https://example.com/1",
                "content": "This is test content for source 1.",
            }
        ]
        result = aggregate_sources(sources)
        assert "SOURCE 1" in result
        assert "Title: Test Article" in result
        assert "URL: https://example.com/1" in result
        assert "Content: This is test content for source 1." in result

    def test_multiple_sources(self):
        """aggregate_sources() formats multiple sources with sequential numbering."""
        sources = [
            {
                "title": "Article One",
                "url": "https://example.com/1",
                "content": "Content one.",
            },
            {
                "title": "Article Two",
                "url": "https://example.com/2",
                "content": "Content two.",
            },
            {
                "title": "Article Three",
                "url": "https://example.com/3",
                "content": "Content three.",
            },
        ]
        result = aggregate_sources(sources)
        assert "SOURCE 1" in result
        assert "SOURCE 2" in result
        assert "SOURCE 3" in result
        assert result.index("SOURCE 1") < result.index("SOURCE 2")
        assert result.index("SOURCE 2") < result.index("SOURCE 3")

    def test_missing_keys_fall_back_to_defaults(self):
        """Missing keys use sensible defaults (Untitled, empty URL, empty content)."""
        sources = [
            {
                "title": "Has All",
                "url": "https://example.com",
                "content": "Full content.",
            },
            {},  # completely empty
        ]
        result = aggregate_sources(sources)
        assert "Title: Has All" in result
        assert "Title: Untitled" in result  # fallback for missing title


class TestBuildFactExtractionContext:
    """Tests for build_fact_extraction_context()."""

    def test_delegates_to_aggregate_sources(self):
        """build_fact_extraction_context produces the same output as aggregate_sources."""
        sources = [
            {
                "title": "Test Article",
                "url": "https://example.com",
                "content": "Test content.",
            }
        ]
        expected = aggregate_sources(sources)
        result = build_fact_extraction_context(sources)
        assert result == expected

    def test_returns_string(self):
        """build_fact_extraction_context() returns a string."""
        sources = [
            {
                "title": "Test",
                "url": "https://example.com",
                "content": "Test content.",
            }
        ]
        result = build_fact_extraction_context(sources)
        assert isinstance(result, str)
        assert len(result) > 0


class TestEstimateContextSize:
    """Tests for estimate_context_size()."""

    def test_returns_character_count(self):
        """estimate_context_size() returns the correct character count."""
        sources = [
            {
                "title": "Test",
                "url": "https://example.com",
                "content": "Hello world.",
            }
        ]
        size = estimate_context_size(sources)
        assert isinstance(size, int)
        assert size > 0

    def test_empty_sources_returns_zero(self):
        """estimate_context_size([]) returns 0."""
        size = estimate_context_size([])
        assert size == 0

    def test_larger_sources_produce_larger_count(self):
        """More/larger sources produce a larger character count."""
        small = [
            {
                "title": "A",
                "url": "https://a.com",
                "content": "Short.",
            }
        ]
        large = [
            {
                "title": "A" * 100,
                "url": "https://a.com/" + "x" * 100,
                "content": "Long content." * 100,
            }
        ]
        small_size = estimate_context_size(small)
        large_size = estimate_context_size(large)
        assert large_size > small_size
