import json


def clean_json_response(text):

    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "")

    if text.startswith("```"):
        text = text.replace("```", "")

    text = text.strip()

    return text


def safe_json_parse(result, fallback=None):

    try:

        result = clean_json_response(result)

        return json.loads(result)

    except Exception as e:

        print(f"[JSON ERROR] {e}")

        return fallback