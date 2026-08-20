from __future__ import annotations

import unittest

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.core.services.generation.classifier import LauncherClassifier


class LauncherClassifierTest(unittest.TestCase):
    """
    Cover package role recognition across known and injected package sets.
    """

    def setUp(self) -> None:
        """
        Build a default launcher classifier.
        """

        self.__classifier = LauncherClassifier()

    def test_recognises_every_launcher_package(self) -> None:
        """
        Each package in the launcher set is classified as a launcher.
        """

        for package in sorted(LAUNCHER_PACKAGES):
            self.assertTrue(self.__classifier.is_launcher(package=package), package)

    def test_rejects_a_real_app_package(self) -> None:
        """
        A real app package is not classified as a launcher.
        """

        self.assertFalse(self.__classifier.is_launcher(package="com.example.shop"))

    def test_honours_an_injected_launcher_set(self) -> None:
        """
        An injected launcher set fully replaces the default set.
        """

        classifier = LauncherClassifier(launchers=frozenset({"com.custom.home"}))

        self.assertTrue(classifier.is_launcher(package="com.custom.home"))
        self.assertFalse(classifier.is_launcher(package=sorted(LAUNCHER_PACKAGES)[0]))
