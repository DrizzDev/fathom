from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PackageClassifier(Protocol):
    """
    Classifies application packages for deterministic trace normalization.
    """

    def is_launcher(self, *, package: str) -> bool:
        """
        Report whether a package is a device launcher / home screen.
        """
        ...
