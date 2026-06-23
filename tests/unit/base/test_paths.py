from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from fathom.base.paths import SharedPathManager
from fathom.schemas.artifact import ArtifactKind
from fathom.settings.env import FathomSettings


class SharedPathManagerLayoutTest(unittest.TestCase):
    """
    Pins the flattened layout {category}/{date}/{session}/{filename}
    """

    def setUp(self) -> None:
        """
        Build a path manager rooted at a temporary directory per test.
        """

        self.__tmp = TemporaryDirectory()
        settings = FathomSettings.model_construct(assets_path=Path(self.__tmp.name))

        self.__manager = SharedPathManager(settings=settings)

    def tearDown(self) -> None:
        """
        Remove the temporary directory between tests.
        """

        self.__tmp.cleanup()

    def __today(self) -> str:
        """
        Build the date stamp the manager uses in its directory layout.
        """

        return datetime.now().strftime("%Y-%m-%d")

    def test_knowledge_db_is_namespaced_per_package(self) -> None:
        """
        A package-scoped knowledge DB lives under memory/knowledge/<package>.db,
        distinct from the shared default and from another package's database.
        """

        shared = self.__manager.get_knowledge_db_path()
        swiggy = self.__manager.get_knowledge_db_path(package="in.swiggy.android")
        zocdoc = self.__manager.get_knowledge_db_path(package="com.zocdoc.android")

        self.assertEqual(shared.name, "knowledge.db")
        self.assertEqual(swiggy.name, "in.swiggy.android.db")
        self.assertEqual(swiggy.parent.name, "knowledge")
        self.assertNotEqual(swiggy, zocdoc)
        self.assertNotEqual(swiggy, shared)

    def test_screenshot_path_has_no_package_level(self) -> None:
        """
        Screenshot path is {root}/screenshot/{date}/{session}/{file}.
        """

        path = self.__manager.get_screenshot_path(session_id="session__1", filename="a.png")

        self.assertEqual(path.parts[-4:], ("screenshot", self.__today(), "session__1", "a.png"))

    def test_trace_path_has_no_package_level(self) -> None:
        """
        Trace path is {root}/traces/{date}/{session}/{file}.
        """

        path = self.__manager.get_trace_path(session_id="session__1", filename="t.png")

        self.assertEqual(path.parts[-4:], ("traces", self.__today(), "session__1", "t.png"))

    def test_xml_path_has_no_package_level(self) -> None:
        """
        XML path is {root}/xmls/{date}/{session}/{file}.
        """

        path = self.__manager.get_xml_path(session_id="session__1", filename="d.xml")

        self.assertEqual(path.parts[-4:], ("xmls", self.__today(), "session__1", "d.xml"))

    def test_annotated_path_has_no_package_level(self) -> None:
        """
        Annotated path is {root}/annotated/{date}/{session}/{file}.
        """

        path = self.__manager.get_annotated_path(session_id="session__1", filename="a.png")

        self.assertEqual(path.parts[-4:], ("annotated", self.__today(), "session__1", "a.png"))

    def test_history_directory_has_no_package_level(self) -> None:
        """
        History directory is {root}/history/{date}/{session}/.
        """

        path = self.__manager.get_history_directory(session_id="session__1")

        self.assertEqual(path.parts[-3:], ("history", self.__today(), "session__1"))

    def test_artifact_path_routes_through_artifact_category(self) -> None:
        """
        get_artifact_path uses the ArtifactCategory mapping and emits the flat layout.
        """

        path = self.__manager.get_artifact_path(
            session_id="session__1",
            kind=ArtifactKind.HIERARCHY_XML,
            filename="dump.xml",
        )

        self.assertEqual(path.parts[-3], self.__today())
        self.assertEqual(path.parts[-2], "session__1")
        self.assertTrue(path.parts[-1].endswith(".xml"))


if __name__ == "__main__":
    unittest.main()
