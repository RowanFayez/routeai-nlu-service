from __future__ import annotations

import json
from typing import Any


def _iter_json_candidates(text: str):
    """Yield substrings that *might* be JSON objects.

    Uses a brace-stack scan to find balanced {...} spans.
    """
    start = None
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]
                    start = None


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from model output."""
    last_error: Exception | None = None

    # Try direct parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        last_error = e

    for cand in _iter_json_candidates(text):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_error = e

    raise ValueError(f"No valid JSON object found. Last error: {last_error}")
