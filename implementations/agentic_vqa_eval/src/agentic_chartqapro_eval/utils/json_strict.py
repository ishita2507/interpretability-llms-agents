"""Strict JSON parsing policy with repair fallback."""

import json
import re
from typing import Any, Optional

from json_repair import repair_json


def parse_strict(
    text: str,
    required_keys: Optional[list[str]] = None,
) -> tuple[dict[str, Any], bool]:
    """
    Parse JSON from text. Returns (parsed_dict, parse_ok).

    parse_ok=True  → clean parse, no repair needed
    parse_ok=False → either repair was needed or parse failed entirely
    """
    text = text.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 1. Direct parse
    try:
        result = json.loads(text)
        if _check_keys(result, required_keys):
            return result, True
        raise ValueError("Missing required keys")
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Extract first JSON block from surrounding text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if _check_keys(result, required_keys):
                return result, True
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Repair fallback
    try:
        repaired = repair_json(text)
        result = json.loads(repaired)
        if _check_keys(result, required_keys):
            return result, False  # parse_ok=False: needed repair
    except Exception:
        pass

    return {}, False


def _check_keys(result: Any, required_keys: Optional[list[str]]) -> bool:
    """Check if result is a dict and contains all required keys (if any)."""
    if not isinstance(result, dict):
        return False
    return not required_keys or all(k in result for k in required_keys)
