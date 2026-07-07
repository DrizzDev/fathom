from __future__ import annotations

from fathom.constants.dialect.drizz import State


class AssertionSubjectNormalizer:
    """
    Normalizes assertion subjects before Drizz state phrases are rendered.
    """

    def normalize(self, *, subject: str, state: State) -> str:
        """
        Return the subject without a duplicate trailing state phrase.
        """

        text = " ".join(subject.split())
        suffix = f" {state.value}"

        if not text.lower().endswith(suffix):
            return text

        normalized = text[: -len(suffix)].strip()
        return normalized or text
