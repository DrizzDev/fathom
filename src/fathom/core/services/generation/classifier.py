from __future__ import annotations

from typing import FrozenSet

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.interfaces.packages import PackageClassifier


class LauncherClassifier(PackageClassifier):
    """
    Classifies packages against a known device-launcher set (Android and iOS by default).
    """

    def __init__(self, *, launchers: FrozenSet[str] = LAUNCHER_PACKAGES) -> None:
        """
        Bind the launcher package set, defaulting to the platform launcher constants.
        """

        self.__launchers = launchers

    def is_launcher(self, *, package: str) -> bool:
        """
        Report whether the package is a known launcher.
        """

        return package in self.__launchers
