from __future__ import annotations

from fathom.constants import VALIDATE_PREFIX


def enforce_validate_prefix(value: str, field_label: str) -> str:
    """Ensure *value* starts with the Validate prefix, raising ValueError otherwise.

    Returns the stripped text on success.
    """
    text = value.strip()
    if not text:
        raise ValueError(f"{field_label} must not be empty.")
    if not text.lower().startswith(VALIDATE_PREFIX):
        raise ValueError(f"{field_label} must start with 'Validate', got: '{text[:30]}...'")
    return text
