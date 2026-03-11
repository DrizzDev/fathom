import tempfile
import unittest
from pathlib import Path

from fathom.base.paths import SharedPathManager
from fathom.infrastructure.storage.local import LocalImageStorage
from fathom.settings.env import FathomSettings


class TestLocalImageStorage(unittest.IsolatedAsyncioTestCase):
    """
    Unit tests for local artifact routing.
    """

    def setUp(self) -> None:
        """
        Create an isolated asset root for each test.
        """

        self.__temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.__temporary_directory.cleanup)

        settings = FathomSettings.model_construct(
            assets_path=Path(self.__temporary_directory.name),
        )
        path_manager = SharedPathManager(settings=settings)
        self.__storage = LocalImageStorage(path_manager=path_manager)

    async def test_save_routes_xml_to_xml_directory(self) -> None:
        """
        Store XML artifacts under the XML asset category.
        """

        location = await self.__storage.save(
            data=b"<root />",
            metadata={
                "category": "xmls",
                "filename": "screen.xml",
                "session_id": "session-123",
                "package_name": "com.example",
            },
        )

        self.assertIn("/xmls/", location)
        self.assertTrue(location.endswith("screen.xml"))

    async def test_save_routes_annotated_image_to_annotated_directory(self) -> None:
        """
        Store annotated artifacts under the annotated asset category.
        """

        location = await self.__storage.save(
            data=b"png",
            metadata={
                "category": "annotated",
                "filename": "screen.png",
                "session_id": "session-123",
                "package_name": "com.example",
            },
        )

        self.assertIn("/annotated/", location)
        self.assertTrue(location.endswith("screen.png"))

    async def test_save_routes_history_artifact_to_history_directory(self) -> None:
        """
        Store history artifacts under the history asset category.
        """

        location = await self.__storage.save(
            data=b"{}",
            metadata={
                "category": "history",
                "filename": "history.json",
                "session_id": "session-123",
                "package_name": "com.example",
            },
        )

        self.assertIn("/history/", location)
        self.assertTrue(location.endswith("history.json"))
