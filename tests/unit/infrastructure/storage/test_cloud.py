from __future__ import annotations

import unittest

from fathom.core.exceptions import VisionError
from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.schemas.configuration import StorageConfiguration


class GCSFilenameContractTest(unittest.IsolatedAsyncioTestCase):
    """Pins that GCS uploads require a canonical filename and reject missing ones."""

    async def test_missing_filename_raises_vision_error(self) -> None:
        """A save call without ``meta.filename`` must fail fast with a VisionError."""

        storage = GCSImageStorage(
            configuration=StorageConfiguration(
                project_id="proj",
                storage_bucket="bucket",
                credentials=None,
            )
        )

        with self.assertRaises(VisionError):
            await storage.save(
                data=b"PNG",
                metadata={
                    "session_id": "sess",
                    "package_name": "pkg",
                    "category": "screenshot",
                },
            )


class GCSContentTypeTest(unittest.TestCase):
    """Pins the content-type resolution against the resolved filename extension."""

    def test_xml_filename_returns_application_xml(self) -> None:
        """XML uploads carry the ``application/xml`` content-type in GCS metadata."""

        content_type = GCSImageStorage._GCSImageStorage__content_type_for(  # type: ignore[attr-defined]
            filename="xmls/2026-06-04/session__1/dump.xml"
        )

        self.assertEqual(content_type, "application/xml")

    def test_png_filename_returns_image_png(self) -> None:
        """PNG uploads carry the ``image/png`` content-type in GCS metadata."""

        content_type = GCSImageStorage._GCSImageStorage__content_type_for(  # type: ignore[attr-defined]
            filename="screenshot/2026-06-04/session__1/shot.png"
        )

        self.assertEqual(content_type, "image/png")

    def test_unknown_extension_returns_octet_stream(self) -> None:
        """Unknown extensions fall back to ``application/octet-stream`` per HTTP convention."""

        content_type = GCSImageStorage._GCSImageStorage__content_type_for(  # type: ignore[attr-defined]
            filename="xmls/2026-06-04/session__1/dump.bin"
        )

        self.assertEqual(content_type, "application/octet-stream")


if __name__ == "__main__":
    unittest.main()
