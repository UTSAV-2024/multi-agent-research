# ==========================================
# CHANGES MADE:
# ==========================================
#
# 1. Added malformed JSON cleanup
# 2. Added trailing comma cleanup
# 3. Added markdown removal improvements
# 4. Added quote normalization
# 5. Added partial JSON recovery
# 6. Added structured logging
# 7. Added raw response preview logging
# 8. Improved fallback reliability
#
# ==========================================

import json
import re

from app.utils.logger import logger


# ==========================================
# CLEAN RAW LLM RESPONSE
# ==========================================

def clean_json_response(text):

    if not text:

        return ""

    text = text.strip()

    # ==========================================
    # REMOVE MARKDOWN WRAPPERS
    # ==========================================

    text = text.replace("```json", "")
    text = text.replace("```", "")

    text = text.strip()

    # ==========================================
    # NORMALIZE QUOTES
    # ==========================================

    text = text.replace("“", "\"")
    text = text.replace("”", "\"")
    text = text.replace("‘", "'")
    text = text.replace("’", "'")

    # ==========================================
    # REMOVE TRAILING COMMAS
    # ==========================================

    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # ==========================================
    # REMOVE INVALID LINE BREAKS
    # ==========================================

    text = text.replace("\n", " ")

    return text.strip()


# ==========================================
# EXTRACT JSON FROM TEXT
# ==========================================

def extract_json(text):

    """
    Extract a JSON object or array from text that may
    contain surrounding explanatory content.

    Handles:
    - Text before/after JSON block
    - Multiple JSON objects (picks first valid one)
    - Nested braces/escaped quotes
    """

    if not text:
        return None

    text = clean_json_response(text)

    # ==========================================
    # Try full parse first
    # ==========================================

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ==========================================
    # Try to find JSON object {}
    # ==========================================

    brace_depth = 0
    json_start = None

    for i, char in enumerate(text):

        if char == "{":

            if json_start is None:
                json_start = i

            brace_depth += 1

        elif char == "}":

            brace_depth -= 1

            if brace_depth == 0 and json_start is not None:

                candidate = text[json_start:i + 1]

                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

                json_start = None

    # ==========================================
    # Try to find JSON array []
    # ==========================================

    bracket_depth = 0
    json_start = None

    for i, char in enumerate(text):

        if char == "[":

            if json_start is None:
                json_start = i

            bracket_depth += 1

        elif char == "]":

            bracket_depth -= 1

            if bracket_depth == 0 and json_start is not None:

                candidate = text[json_start:i + 1]

                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

                json_start = None

    logger.warning(
        "[EXTRACT JSON] No valid JSON found in response"
    )

    return None


# ==========================================
# SAFE JSON PARSER
# ==========================================

def safe_json_parse(result, fallback=None):

    try:

        result = clean_json_response(result)

        return json.loads(result)

    except Exception as e:

        logger.error(f"[JSON ERROR] {e}")

        logger.warning(
            f"[JSON ERROR] Raw response preview: {result[:300]}"
        )

        # ==========================================
        # ATTEMPT PARTIAL JSON RECOVERY
        # ==========================================

        try:

            start = result.find("{")
            end = result.rfind("}")

            if start != -1 and end != -1:

                partial = result[start:end+1]

                return json.loads(partial)

        except Exception as inner_error:

            logger.error(
                f"[JSON RECOVERY FAILED] {inner_error}"
            )

        return fallback