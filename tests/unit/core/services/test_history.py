from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import Mock

from fathom.base.paths import SharedPathManager
from fathom.core.services.history import HistoryService
from fathom.settings.env import FathomSettings


class HistoryArtifactPathTest(unittest.TestCase):
    """History artifacts stay flat, session-scoped files even when given an activity."""

    def setUp(self) -> None:
        """
        Build a history service rooted at a temporary assets directory.
        """

        self.__tmp = TemporaryDirectory()
        path_manager = SharedPathManager(
            settings=FathomSettings.model_construct(assets_path=Path(self.__tmp.name))
        )
        self.__service = HistoryService(
            workflow_id="session__1",
            package_name="com.app",
            exporter=Mock(),
            path_manager=path_manager,
        )

    def tearDown(self) -> None:
        """
        Remove the temporary assets directory between tests.
        """

        self.__tmp.cleanup()

    def __path(self, *, package_name: str, filename: str) -> Path:
        """
        Resolve a history artifact path through the service's private builder.
        """

        return cast(
            "Path",
            self.__service._HistoryService__get_history_file_path(  # type: ignore[attr-defined]
                package_name=package_name, filename=filename
            ),
        )

    def test_resolve_package_name_drops_the_activity_component(self) -> None:
        resolved = self.__service._HistoryService__resolve_package_name(  # type: ignore[attr-defined]
            package_name="com.app/com.app.activities.SearchActivity"
        )

        self.assertEqual(resolved, "com.app")

    def test_history_path_never_nests_on_a_slash(self) -> None:
        # Defensive layer: even a raw activity must stay a flat, single file.
        path = self.__path(
            package_name="com.app/com.app.activities.SearchActivity", filename="history.json"
        )

        self.assertNotIn("/", path.name)
        self.assertEqual(path.parent.name, "session__1")

    def test_resolved_package_yields_a_single_history_file(self) -> None:
        resolved = self.__service._HistoryService__resolve_package_name(  # type: ignore[attr-defined]
            package_name="com.app/com.app.activities.SearchActivity"
        )
        path = self.__path(package_name=resolved, filename="history.json")

        self.assertEqual(path.name, "history__com.app.json")

    def test_history_path_embeds_a_plain_package(self) -> None:
        path = self.__path(package_name="com.app", filename="history.json")

        self.assertEqual(path.name, "history__com.app.json")


if __name__ == "__main__":
    unittest.main()
