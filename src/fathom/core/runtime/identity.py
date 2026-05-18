from __future__ import annotations


class TargetIdentity:
    """
    Pure normalization helpers for comparing semantic target descriptions.
    """

    @staticmethod
    def normalize(*, description: str) -> str:
        """
        Reduce a description to its canonical surface form for equality checks.
        """

        return " ".join(description.lower().split()).rstrip(".!?")

    @classmethod
    def describes_same_target(cls, *, previous: str, replacement: str) -> bool:
        """
        Return whether two descriptions point at the same imperative target.
        """

        return cls.normalize(description=previous) == cls.normalize(description=replacement)
