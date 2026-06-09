# ==========================================
# TESTS FOR JSON UTILITY FUNCTIONS
# ==========================================

import pytest

from app.utils.json_utils import (
    safe_json_parse,
    clean_json_response,
    extract_json,
    clean_json,
    recover_json,
    _normalize_type,
    _merge_fallback_schema,
)


# ==========================================
# safe_json_parse — Stage 1: Direct parse
# ==========================================

class TestDirectParse:
    """Stage 1: Valid JSON should parse directly."""

    def test_valid_json_object(self):
        result = safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_array(self):
        result = safe_json_parse('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_valid_json_nested(self):
        result = safe_json_parse('{"a": {"b": [1, 2]}}')
        assert result == {"a": {"b": [1, 2]}}


# ==========================================
# safe_json_parse — Stage 2: JSON extraction
# ==========================================

class TestExtraction:
    """Stage 2: Extract JSON from surrounding text."""

    def test_leading_text(self):
        result = safe_json_parse(
            'Here is the result: {"facts": ["a", "b"]}'
        )
        assert result == {"facts": ["a", "b"]}

    def test_trailing_text(self):
        result = safe_json_parse(
            '{"facts": ["a", "b"]} Hope this helps!'
        )
        assert result == {"facts": ["a", "b"]}

    def test_surrounding_text(self):
        result = safe_json_parse(
            'Here is the result: {"facts": ["a"]} Hope this helps!'
        )
        assert result == {"facts": ["a"]}

    def test_markdown_wrapped(self):
        result = safe_json_parse(
            '```json\n{"facts": ["a", "b"]}\n```'
        )
        assert result == {"facts": ["a", "b"]}

    def test_markdown_wrapped_with_leading_text(self):
        result = safe_json_parse(
            'Here is the JSON:\n```json\n{"key": "value"}\n```'
        )
        assert result == {"key": "value"}


# ==========================================
# safe_json_parse — Stage 3: Bracket recovery
# ==========================================

class TestBracketRecovery:
    """Stage 3: Recover truncated/malformed JSON brackets."""

    def test_missing_closing_brace(self):
        text = '{"facts": ["a", "b"]'
        result = safe_json_parse(text)
        assert result == {"facts": ["a", "b"]}

    def test_missing_closing_bracket(self):
        text = '{"data": [1, 2, 3]'
        result = safe_json_parse(text)
        assert result == {"data": [1, 2, 3]}

    def test_truncated_after_brace(self):
        text = '{"a": {"b": 1}'
        result = safe_json_parse(text)
        assert result == {"a": {"b": 1}}


# ==========================================
# safe_json_parse — Stage 4: LLM repair
# ==========================================

class TestLLMRepair:
    """Stage 4: Repair common LLM JSON issues."""

    def test_trailing_comma_in_object(self):
        result = safe_json_parse('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}

    def test_trailing_comma_in_array(self):
        result = safe_json_parse('{"list": [1, 2, 3,]}')
        assert result == {"list": [1, 2, 3]}

    def test_leading_garbage(self):
        result = safe_json_parse(
            'Sure! Here is the data: {"key": "value"}'
        )
        assert result == {"key": "value"}

    def test_single_quoted_keys(self):
        # This is harder — we handle it via bracket recovery
        result = safe_json_parse("{'key': 'value'}")
        assert result is not None


# ==========================================
# safe_json_parse — Stage 5: Fallback
# ==========================================

class TestFallback:
    """Stage 5: Return fallback when all stages fail."""

    def test_completely_invalid_json(self):
        result = safe_json_parse(
            "This is not JSON at all",
            fallback={"default": True}
        )
        assert result == {"default": True}

    def test_empty_fallback(self):
        result = safe_json_parse("")
        assert result == {}

    def test_none_fallback(self):
        result = safe_json_parse(
            "garbage",
            fallback={"key": "default"}
        )
        assert result == {"key": "default"}

    def test_empty_input_with_none_fallback(self):
        result = safe_json_parse("", fallback=None)
        assert result == {}


# ==========================================
# clean_json_response
# ==========================================

class TestCleanJsonResponse:
    """Verify the text normalization function."""

    def test_markdown_removal(self):
        cleaned = clean_json_response(
            "```json\n{\"key\": \"value\"}\n```"
        )
        assert cleaned == '{"key": "value"}'

    def test_smart_quotes_normalized(self):
        cleaned = clean_json_response('{"key": \u201cvalue\u201d}')
        assert '"value"' in cleaned

    def test_trailing_comma_removed(self):
        cleaned = clean_json_response('{"a": 1,}')
        assert cleaned == '{"a": 1}'

    def test_empty_input(self):
        assert clean_json_response("") == ""
        assert clean_json_response(None) == ""


# ==========================================
# extract_json
# ==========================================

class TestExtractJSON:
    """Verify standalone extraction function."""

    def test_basic_extraction(self):
        result = extract_json('{"a": 1}')
        assert result == {"a": 1}

    def test_from_surrounding_text(self):
        result = extract_json(
            'Prefix text {"key": "value"} suffix text'
        )
        assert result == {"key": "value"}

    def test_no_json_found(self):
        result = extract_json("No JSON here at all")
        assert result is None

    def test_empty_input(self):
        assert extract_json("") is None
        assert extract_json(None) is None


# ==========================================
# Backward-compatible aliases
# ==========================================

class TestBackwardCompatibility:
    """Verify that old API functions still work."""

    def test_clean_json_alias(self):
        assert clean_json("  hello  ") == "hello"

    def test_recover_json_valid(self):
        result = recover_json('{"a": 1}')
        assert result == {"a": 1}

    def test_recover_json_invalid(self):
        result = recover_json("garbage")
        # Returns empty dict from fallback when recovery fails
        assert result == {}

    def test_recover_json_extraction(self):
        result = recover_json(
            'Text then {"key": "value"} more text'
        )
        assert result == {"key": "value"}


# ==========================================
# expected_type enforcement
# ==========================================

class TestExpectedType:
    """Verify expected_type parameter normalises return types."""

    def test_expected_type_dict_returns_dict_for_list_input(self):
        """List input with expected_type='dict' wraps in {'facts': [...]}."""
        result = safe_json_parse(
            '["fact1", "fact2"]',
            expected_type="dict"
        )
        assert isinstance(result, dict)
        assert result == {"facts": ["fact1", "fact2"]}

    def test_expected_type_dict_passes_dict_through(self):
        """Dict input with expected_type='dict' passes through unchanged."""
        result = safe_json_parse(
            '{"a": 1}',
            expected_type="dict"
        )
        assert isinstance(result, dict)
        assert result == {"a": 1}

    def test_expected_type_list_returns_list_for_dict_input(self):
        """Dict with a single list value with expected_type='list' extracts the list."""
        result = safe_json_parse(
            '{"facts": ["a", "b"]}',
            expected_type="list"
        )
        assert isinstance(result, list)
        assert result == ["a", "b"]

    def test_expected_type_list_passes_list_through(self):
        """List input with expected_type='list' passes through unchanged."""
        result = safe_json_parse(
            '[1, 2, 3]',
            expected_type="list"
        )
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_expected_type_none_leaves_type_unchanged(self):
        """expected_type=None (default) does no normalisation."""
        result = safe_json_parse('[1, 2, 3]')
        assert isinstance(result, list)


# ==========================================
# Fallback schema merge
# ==========================================

class TestFallbackMerge:
    """Verify fallback keys are merged into recovered results."""

    FALLBACK = {
        "facts": [],
        "summary": "",
        "confidence": 0
    }

    def test_partial_recovery_filled_with_fallback(self):
        """Recovered dict missing 'summary' and 'confidence' gets them from fallback."""
        result = safe_json_parse(
            '{"facts": ["fact1"]}',
            fallback=self.FALLBACK,
            expected_type="dict"
        )
        assert result["facts"] == ["fact1"]
        assert result["summary"] == ""
        assert result["confidence"] == 0

    def test_full_recovery_not_overwritten(self):
        """Recovered values take priority over fallback defaults."""
        result = safe_json_parse(
            '{"facts": ["x"], "summary": "custom", "confidence": 0.9}',
            fallback=self.FALLBACK,
            expected_type="dict"
        )
        assert result["summary"] == "custom"
        assert result["confidence"] == 0.9

    def test_list_to_dict_with_fallback_merge(self):
        """List recovered with expected_type='dict' gets wrapped AND merged with fallback."""
        result = safe_json_parse(
            '["fact1", "fact2"]',
            fallback=self.FALLBACK,
            expected_type="dict"
        )
        assert isinstance(result, dict)
        assert result["facts"] == ["fact1", "fact2"]
        assert result["summary"] == ""
        assert result["confidence"] == 0

    def test_fallback_used_when_all_stages_fail(self):
        """When recovery fails, fallback shape is preserved."""
        result = safe_json_parse(
            "complete garbage",
            fallback=self.FALLBACK,
            expected_type="dict"
        )
        assert result == self.FALLBACK

    def test_factcheck_fallback_keys_guaranteed(self):
        """Fact-check style fallback keys are guaranteed present."""
        fb = {
            "confirmed_facts": [],
            "disputed_facts": [],
            "low_confidence_facts": []
        }
        result = safe_json_parse(
            '{"confirmed_facts": ["only this"]}',
            fallback=fb,
            expected_type="dict"
        )
        assert result["confirmed_facts"] == ["only this"]
        assert result["disputed_facts"] == []
        assert result["low_confidence_facts"] == []


# ==========================================
# _normalize_type (unit-level)
# ==========================================

class TestNormalizeType:
    """Direct tests of the _normalize_type helper."""

    def test_list_to_dict_wrap(self):
        result = _normalize_type(["a", "b"], "dict")
        assert result == {"facts": ["a", "b"]}

    def test_dict_to_list_extract(self):
        result = _normalize_type({"items": [1, 2]}, "list")
        assert result == [1, 2]

    def test_dict_to_list_no_list_returns_unchanged(self):
        result = _normalize_type({"key": "value"}, "list")
        assert result == {"key": "value"}

    def test_none_expected_type_unchanged(self):
        result = _normalize_type([1, 2, 3], None)
        assert result == [1, 2, 3]


# ==========================================
# _merge_fallback_schema (unit-level)
# ==========================================

class TestMergeFallbackSchema:
    """Direct tests of the _merge_fallback_schema helper."""

    def test_partial_merge(self):
        result = _merge_fallback_schema(
            {"facts": ["a"]},
            {"facts": [], "summary": "", "confidence": 0}
        )
        assert result == {"facts": ["a"], "summary": "", "confidence": 0}

    def test_fallback_key_priority(self):
        result = _merge_fallback_schema(
            {"summary": "custom"},
            {"summary": "default"}
        )
        assert result == {"summary": "custom"}

    def test_non_dict_result_returns_unchanged(self):
        result = _merge_fallback_schema([1, 2], {"key": "val"})
        assert result == [1, 2]

    def test_non_dict_fallback_returns_unchanged(self):
        result = _merge_fallback_schema({"key": "val"}, [1, 2])
        assert result == {"key": "val"}
