# ==========================================
# MULTI-STAGE JSON RECOVERY PIPELINE
# ==========================================
#
# Provides robust JSON parsing for LLM outputs
# that are frequently malformed.
#
# Public API:
#   safe_json_parse(text, fallback)    - 5-stage recovery pipeline
#   clean_json_response(text)          - pre-clean / normalize text
#   extract_json(text)                 - extract first valid JSON
#   clean_json(text)                   - backward-compatible alias
#   recover_json(text)                 - backward-compatible alias
#
# ==========================================

import json
import re

from app.utils.logger import logger


# ==========================================
# STAGE 0 - TEXT CLEANUP / NORMALIZATION
# ==========================================

def clean_json_response(text: str) -> str:
    """
    Pre-clean raw LLM output before JSON parsing.

    Handles:
    - Markdown code fences (```json / ```)
    - Smart/curly quotes
    - Trailing commas in objects and arrays
    - Invisible unicode characters
    """
    if not text:
        return ""

    text = text.strip()

    # --- Remove markdown wrappers ---
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    # --- Normalize smart quotes ---
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")

    # --- Remove trailing commas ---
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # --- Remove null bytes and other invisible chars ---
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


# ==========================================
# STAGE 1 - DIRECT JSON PARSE
# ==========================================

def _stage1_direct_parse(text: str):
    """Attempt direct json.loads() after cleaning."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ==========================================
# STAGE 2 - EXTRACT JSON FROM SURROUNDING TEXT
# ==========================================

def _find_first_json_value(text: str):
    """
    Find the first complete JSON value (object or array)
    within arbitrary surrounding text, using bracket matching.

    Tracks BOTH bracket types ({ and [) simultaneously so that
    inner arrays inside unclosed outer objects are NOT returned
    as "complete" values — they fall through to Stage 3 recovery.
    """
    depth_obj = 0
    depth_arr = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            if start is None:
                start = i
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1

        if depth_obj == 0 and depth_arr == 0 and start is not None:
            # All brackets are balanced — this is a complete JSON value
            candidate = text[start:i + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                start = None
    return None


def _stage2_extract_json(text: str):
    """Try to extract the first valid JSON value from surrounding text."""
    return _find_first_json_value(text)


# ==========================================
# STAGE 3 - BRACKET MATCHING RECOVERY
# ==========================================

def _try_balanced_braces(text: str):
    """
    Attempt to balance braces/brackets by truncating or
    appending missing closing characters.
    """
    for start_idx in range(len(text)):
        if text[start_idx] != "{":
            continue
        for end_idx in range(start_idx + 1, len(text) + 1):
            if end_idx < len(text) and text[end_idx] not in ("}", ","):
                continue
            candidate = text[start_idx:end_idx]
            open_count = candidate.count("{")
            close_count = candidate.count("}")
            if open_count > close_count:
                candidate += "}" * (open_count - close_count)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    for start_idx in range(len(text)):
        if text[start_idx] != "[":
            continue
        for end_idx in range(start_idx + 1, len(text) + 1):
            if end_idx < len(text) and text[end_idx] not in ("]", ","):
                continue
            candidate = text[start_idx:end_idx]
            open_count = candidate.count("[")
            close_count = candidate.count("]")
            if open_count > close_count:
                candidate += "]" * (open_count - close_count)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    return None


def _stage3_bracket_recovery(text: str):
    """Bracket matching recovery for truncated JSON."""
    return _try_balanced_braces(text)


# ==========================================
# STAGE 4 - COMMON LLM REPAIR
# ==========================================

def _llm_repair(text: str):
    """
    Apply various heuristics to repair common LLM JSON issues.
    """
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_brace == -1 and first_bracket == -1:
        return None
    if first_brace == -1:
        start = first_bracket
    elif first_bracket == -1:
        start = first_brace
    else:
        start = min(first_brace, first_bracket)
    text = text[start:]

    last_valid_end = -1
    depth_obj = 0
    depth_arr = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj -= 1
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr -= 1
        if depth_obj == 0 and depth_arr == 0 and i > 0:
            last_valid_end = i

    if last_valid_end > 0:
        text = text[:last_valid_end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    def _count_unescaped_quotes(s: str) -> int:
        """Count double-quote chars that are NOT preceded by a backslash."""
        count = 0
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s) and s[i + 1] == '"':
                i += 2  # skip escaped quote
                continue
            if s[i] == '"':
                count += 1
            i += 1
        return count

    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        stripped = line.rstrip()
        if _count_unescaped_quotes(stripped) % 2 != 0:
            last_open = stripped.rfind('"')
            if last_open > 0 and stripped[last_open - 1] != "\\":
                before_quote = stripped[:last_open]
                if ":" in before_quote:
                    stripped = before_quote
                else:
                    stripped = stripped + '"'
        fixed_lines.append(stripped)
    text = "\n".join(fixed_lines)

    text = text.strip()
    open_braces = text.count("{") - text.count("}")
    if open_braces > 0:
        text += "}" * open_braces
    open_brackets = text.count("[") - text.count("]")
    if open_brackets > 0:
        text += "]" * open_brackets

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(
        r'(?<!")(\b[a-zA-Z_][a-zA-Z0-9_]*\b)(?=\s*:)',
        r'"\1"',
        text
    )
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


def _stage4_llm_repair(text: str):
    """Apply common LLM JSON repair logic."""
    return _llm_repair(text)


# ==========================================
# TYPE NORMALIZATION HELPERS
# ==========================================

def _normalize_type(
    result: Any,
    expected_type: str | None
) -> Any:
    """
    Normalize the recovered result to match the expected type.

    - If expected_type="dict" and result is a list, wrap it as {"facts": list}.
    - If expected_type="list" and result is a dict with a single list value, extract it.
    - Otherwise return result unchanged.
    """
    if expected_type is None:
        return result

    if expected_type == "dict" and isinstance(result, list):
        logger.info(
            "[JSON PARSER] Recovered list wrapped into dict"
        )
        return {"facts": result}

    if expected_type == "list" and isinstance(result, dict):
        # Try to extract the first list value from the dict
        for val in result.values():
            if isinstance(val, list):
                return val
        # No list found — return as-is (caller expects list but got dict)
        return result

    return result


def _merge_fallback_schema(
    result: dict,
    fallback: dict | None
) -> dict:
    """
    Ensure all keys from the fallback dict exist in the result.
    Missing keys are filled with their fallback values.

    Only applies when both result and fallback are dicts.
    """
    if not isinstance(result, dict) or not isinstance(fallback, dict):
        return result

    merged = dict(fallback)  # start with fallback keys
    merged.update(result)    # overlay recovered values

    logger.info(
        "[JSON PARSER] Fallback schema merged"
    )
    return merged


# ==========================================
# MAIN PIPELINE - safe_json_parse
# ==========================================

def safe_json_parse(
    text: str,
    fallback: dict | list | None = None,
    expected_type: str | None = None
) -> dict | list:
    """
    Central JSON parsing function with multi-stage recovery.

    Attempts recovery in the following order:
    Stage 1: Direct json.loads()
    Stage 2: Extract first valid JSON from surrounding text
    Stage 3: Bracket matching recovery
    Stage 4: Common LLM repair logic
    Stage 5: Return fallback (with shape preservation)

    After recovery, normalises the return type if expected_type
    is specified, and merges fallback keys into the result so
    all expected keys are guaranteed present.

    Args:
        text: Raw text to parse (typically LLM output).
        fallback: Default return value if all stages fail.
        expected_type: "dict" or "list" to enforce return type.
                       None means no type enforcement.

    Returns:
        Parsed dict/list, or fallback (default {}).
    """
    if not text:
        logger.warning("[JSON PARSER] Empty input, using fallback")
        result = fallback or {}
        if isinstance(result, dict) and isinstance(fallback, dict):
            result = _merge_fallback_schema(result, fallback)
        return result

    cleaned = clean_json_response(text)

    # Stage 1 - Direct parse
    result = _stage1_direct_parse(cleaned)
    if result is not None:
        logger.info("[JSON PARSER] Direct parse successful")
        return _finalize_result(result, expected_type, fallback)

    # Stage 2 - Extract from surrounding text
    result = _stage2_extract_json(cleaned)
    if result is not None:
        logger.info("[JSON PARSER] JSON extraction successful")
        return _finalize_result(result, expected_type, fallback)

    # Stage 3 - Bracket matching recovery
    result = _stage3_bracket_recovery(cleaned)
    if result is not None:
        logger.info("[JSON PARSER] Bracket recovery successful")
        return _finalize_result(result, expected_type, fallback)

    # Stage 4 - Common LLM repair
    result = _stage4_llm_repair(cleaned)
    if result is not None:
        logger.info("[JSON PARSER] Repair recovery successful")
        return _finalize_result(result, expected_type, fallback)

    # Stage 5 - Return fallback
    logger.warning(
        "[JSON PARSER] All recovery stages failed for input, using fallback"
    )
    logger.debug(
        f"[JSON PARSER] Raw input preview: {text[:300]}"
    )
    if fallback is not None:
        if isinstance(fallback, dict):
            result = _merge_fallback_schema(fallback, fallback)
        else:
            result = fallback
    else:
        result = {}
    return result


def _finalize_result(
    result: Any,
    expected_type: str | None,
    fallback: dict | list | None
) -> Any:
    """Apply type normalisation and fallback-schema merge to a recovered result."""
    # 1. Normalise type
    result = _normalize_type(result, expected_type)

    # 2. Merge fallback schema if result is a dict and fallback is a dict
    if isinstance(result, dict) and isinstance(fallback, dict):
        # If type normalisation wrapped a list into {"facts": [...]} and the
        # fallback has more specific list-typed keys (e.g. "low_confidence_facts"),
        # migrate the wrapped data to the first matching fallback list key.
        if "facts" in result and len(result) == 1:
            for key, val in fallback.items():
                if isinstance(val, list) and key != "facts":
                    result[key] = result.pop("facts")
                    break

        result = _merge_fallback_schema(result, fallback)

    # ==========================================
    # DIAGNOSTIC: Log final result structure
    # ==========================================

    logger.info(
        "[JSON FINALIZE] keys=%s",
        list(result.keys()) if isinstance(result, dict) else type(result).__name__
    )

    if isinstance(result, dict) and "facts" in result:
        logger.warning(
            "[JSON FINALIZE] facts key contains %s items",
            len(result.get("facts", []))
        )

    logger.info(
        f"[JSON PARSER] Returned type={type(result).__name__}"
    )
    return result


# ==========================================
# BACKWARD-COMPATIBLE FUNCTIONS
# ==========================================

def clean_json(text: str) -> str:
    """Backward-compatible alias for clean_json_response()."""
    return clean_json_response(text)


def recover_json(text: str) -> dict | list | None:
    """
    Backward-compatible alias that attempts full recovery.

    Returns the parsed JSON if successful, or None if all
    recovery stages fail.
    """
    result = safe_json_parse(text, fallback=None)
    return result


# ==========================================
# EXTRACT JSON - standalone extraction
# ==========================================

def extract_json(text: str) -> dict | list | None:
    """
    Extract the first valid JSON object or array from text
    that may contain surrounding explanatory content.

    Handles:
    - Text before/after JSON block
    - Multiple JSON objects (picks first valid one)
    - Nested braces and escaped quotes

    Returns:
        Parsed JSON value, or None if no valid JSON found.
    """
    if not text:
        return None

    text = clean_json_response(text)

    # Try full parse first
    result = _stage1_direct_parse(text)
    if result is not None:
        return result

    # Try extraction from surrounding text
    result = _stage2_extract_json(text)
    if result is not None:
        return result

    # Try bracket recovery
    result = _stage3_bracket_recovery(text)
    if result is not None:
        return result

    logger.warning(
        "[EXTRACT JSON] No valid JSON found in response"
    )
    return None
