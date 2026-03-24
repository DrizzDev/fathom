from __future__ import annotations


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    result = text.strip()
    if result.startswith("```json"):
        result = result[7:]
    elif result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]
    return result.strip()
