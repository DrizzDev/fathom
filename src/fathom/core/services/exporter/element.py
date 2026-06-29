"""
Normalisation of a link's element target toward its exact on-screen visible text.

The vision model describes a tap target by its visible text plus a generic
element-type descriptor (for example "Continue button"). The test-authoring
consumer grounds on the exact visible text, so the descriptor is removed.
"""

from __future__ import annotations

from fathom.constants.document import GENERIC_ELEMENT_SUFFIXES, MINIMUM_VISIBLE_TARGET_LENGTH


class ElementText:
    """
    Reduces a vision-model element target to the element's exact visible text.
    """

    @classmethod
    def visible(cls, *, target: str) -> str:
        """
        Strips a single trailing generic element-type descriptor from a target.
        """

        cleaned = target.strip()
        lowered = cleaned.lower()
        for suffix in GENERIC_ELEMENT_SUFFIXES:
            token = f" {suffix}"
            if lowered.endswith(token):
                remainder = cleaned[: len(cleaned) - len(token)].strip()
                return remainder if cls.__is_meaningful(text=remainder) else cleaned
        return cleaned

    @staticmethod
    def __is_meaningful(*, text: str) -> bool:
        """
        Whether the stripped remainder is still a usable visible-text target.
        """

        return len(text) >= MINIMUM_VISIBLE_TARGET_LENGTH and any(
            character.isalnum() for character in text
        )
