from __future__ import annotations

from typing import FrozenSet

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.interfaces.packages import PackageClassifier


class LauncherClassifier(PackageClassifier):
    """
    Classifies packages used by deterministic trace normalization.
    """

    def __init__(self, *, launchers: FrozenSet[str] = LAUNCHER_PACKAGES) -> None:
        """
        Bind package role sets used by trace normalization.
        """

        self.__launchers = launchers

    def is_launcher(self, *, package: str) -> bool:
        """
        Report whether the package is a known launcher.
        """

        return package in self.__launchers
