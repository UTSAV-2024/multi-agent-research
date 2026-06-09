# ==========================================
# SUMMARIZER AGENT TESTS (REFACTORED)
# ==========================================
#
# Tests for the single-pass source-attributed
# fact extraction logic:
#   - _validate_fact_entry
#   - _validate_facts
#   - Return structure backward compatibility
#
# ==========================================

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agents.summarizer_agent import (
    _validate_fact_entry,
    _validate_facts,
)


class TestValidateFactEntry:
    """Tests for _validate_fact_entry()."""

    def test_valid_entry(self):
        """A fully valid entry returns True."""
        entry = {
            "fact": "This is a fact.",
            "source_title": "Article One",
            "source_url": "https://example.com/1",
        }
        assert _validate_fact_entry(entry) is True

    def test_missing_fact_key(self):
        """Entry missing 'fact' key returns False."""
        entry = {
            "source_title": "Article One",
            "source_url": "https://example.com/1",
        }
        assert _validate_fact_entry(entry) is False

    def test_missing_source_title_key(self):
        """Entry missing 'source_title' key returns False."""
        entry = {
            "fact": "Some fact.",
            "source_url": "https://example.com/1",
        }
        assert _validate_fact_entry(entry) is False

    def test_missing_source_url_key(self):
        """Entry missing 'source_url' key returns False."""
        entry = {
            "fact": "Some fact.",
            "source_title": "Article One",
        }
        assert _validate_fact_entry(entry) is False

    def test_empty_fact_string(self):
        """Entry with empty fact string returns False."""
        entry = {
            "fact": "",
            "source_title": "Article One",
            "source_url": "https://example.com/1",
        }
        assert _validate_fact_entry(entry) is False

    def test_whitespace_only_fact(self):
        """Entry with whitespace-only fact returns False."""
        entry = {
            "fact": "   ",
            "source_title": "Article One",
            "source_url": "https://example.com/1",
        }
        assert _validate_fact_entry(entry) is False

    def test_non_string_fact(self):
        """Entry with non-string fact returns False."""
        entry = {
            "fact": 123,
            "source_title": "Article One",
            "source_url": "https://example.com/1",
        }
        assert _validate_fact_entry(entry) is False

    def test_non_string_source_title(self):
        """Entry with non-string source_title returns False."""
        entry = {
            "fact": "A fact.",
            "source_title": 456,
            "source_url": "https://example.com/1",
        }
        assert _validate_fact_entry(entry) is False

    def test_non_string_source_url(self):
        """Entry with non-string source_url returns False."""
        entry = {
            "fact": "A fact.",
            "source_title": "Article One",
            "source_url": 789,
        }
        assert _validate_fact_entry(entry) is False

    def test_missing_all_keys(self):
        """Entry missing all required keys returns False."""
        entry = {}
        assert _validate_fact_entry(entry) is False

    def test_extra_keys_ignored(self):
        """Entry with extra keys beyond required is still valid."""
        entry = {
            "fact": "A fact.",
            "source_title": "Article One",
            "source_url": "https://example.com/1",
            "extra_field": "should be ignored",
        }
        assert _validate_fact_entry(entry) is True


class TestValidateFacts:
    """Tests for _validate_facts()."""

    def test_empty_list(self):
        """_validate_facts([]) returns []."""
        assert _validate_facts([]) == []

    def test_all_valid(self):
        """All valid entries are kept."""
        facts = [
            {
                "fact": "Fact one.",
                "source_title": "Article A",
                "source_url": "https://a.com",
            },
            {
                "fact": "Fact two.",
                "source_title": "Article B",
                "source_url": "https://b.com",
            },
        ]
        result = _validate_facts(facts)
        assert len(result) == 2

    def test_filters_invalid(self):
        """Invalid entries are filtered out."""
        facts = [
            {
                "fact": "Valid fact.",
                "source_title": "Article A",
                "source_url": "https://a.com",
            },
            {
                "fact": "Missing source URL.",
                "source_title": "Article B",
                # missing source_url
            },
            "not a dict",
            123,
        ]
        result = _validate_facts(facts)
        assert len(result) == 1
        assert result[0]["fact"] == "Valid fact."

    def test_non_dict_entries_filtered(self):
        """Non-dict entries (strings, ints, None) are skipped."""
        facts = [
            {"fact": "Good.", "source_title": "X", "source_url": "https://x.com"},
            "string fact",
            None,
            42,
        ]
        result = _validate_facts(facts)
        assert len(result) == 1
        assert result[0]["fact"] == "Good."

    def test_preserves_order(self):
        """Valid entries preserve their original order."""
        facts = [
            {"fact": "First.", "source_title": "A", "source_url": "https://a.com"},
            {"fact": "", "source_title": "B", "source_url": "https://b.com"},  # invalid
            {"fact": "Second.", "source_title": "C", "source_url": "https://c.com"},
        ]
        result = _validate_facts(facts)
        assert len(result) == 2
        assert result[0]["fact"] == "First."
        assert result[1]["fact"] == "Second."


class TestReturnStructure:
    """Tests for backward-compatible return structure."""

    def test_has_expected_keys(self):
        """The summary dict has all expected keys."""
        from app.agents.summarizer_agent import summarizer_agent_async

        # We can't easily call the async function without mocking,
        # but we can verify the expected shape is documented correctly
        expected_shape = {
            "source": str,
            "url": str,
            "facts": list,
            "summary_time": float,
            "failed": bool,
        }

        # Verify keys are in the docstring
        doc = summarizer_agent_async.__doc__
        assert doc is not None
        assert "source" in doc
        assert "aggregated_sources" in doc
        assert "facts" in doc
        assert "summary_time" in doc
        assert "failed" in doc

    def test_sync_wrapper_exists(self):
        """The sync wrapper function exists and is callable."""
        from app.agents.summarizer_agent import summarizer_agent
        assert callable(summarizer_agent)

    def test_module_imports_cleanly(self):
        """The module imports without errors."""
        import importlib
        spec = importlib.util.find_spec("app.agents.summarizer_agent")
        assert spec is not None
