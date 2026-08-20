from __future__ import annotations

import unittest

from fathom.infrastructure.storage.cloud import GCSImageStorage


class GCSFilenameFallbackTest(unittest.TestCase):
    """
    Pins generated filenames for direct storage callers.
    """

    def test_xmls_category_uses_xml_extension(self) -> None:
        """
        Hierarchy XML uploads without a caller filename still land as XML.
        """

        filename = GCSImageStorage._GCSImageStorage__fallback_filename(  # type: ignore[attr-defined]
            category="xmls",
            activity="MainActivity",
            package="com.example",
        )

        self.assertTrue(filename.endswith("__com.example__MainActivity.xml"))

    def test_history_category_uses_text_extension(self) -> None:
        """
        History uploads without a caller filename still land as text.
        """

        filename = GCSImageStorage._GCSImageStorage__fallback_filename(  # type: ignore[attr-defined]
            category="history",
            activity="MainActivity",
            package="com.example",
        )

        self.assertTrue(filename.endswith("__com.example__MainActivity.txt"))

    def test_image_categories_default_to_png_extension(self) -> None:
        """
        Image-bearing categories without a caller filename still land as PNG.
        """

        for category in ("screenshot", "annotated", "traces"):
            with self.subTest(category=category):
                filename = GCSImageStorage._GCSImageStorage__fallback_filename(  # type: ignore[attr-defined]
                    category=category,
                    activity="MainActivity",
                    package="com.example",
                )

                self.assertTrue(filename.endswith("__com.example__MainActivity.png"))


class GCSContentTypeTest(unittest.TestCase):
    """
    Pins the content-type resolution against the resolved filename extension.
    """

    def test_xml_filename_returns_application_xml(self) -> None:
        """
        XML uploads carry the ``application/xml`` content-type in GCS metadata.
        """

        content_type = GCSImageStorage._GCSImageStorage__content_type_for(  # type: ignore[attr-defined]
            filename="xmls/2026-06-04/session__1/dump.xml"
        )

        self.assertEqual(content_type, "application/xml")

    def test_png_filename_returns_image_png(self) -> None:
        """
        PNG uploads carry the ``image/png`` content-type in GCS metadata.
        """

        content_type = GCSImageStorage._GCSImageStorage__content_type_for(  # type: ignore[attr-defined]
            filename="screenshot/2026-06-04/session__1/shot.png"
        )

        self.assertEqual(content_type, "image/png")

    def test_unknown_extension_returns_octet_stream(self) -> None:
        """
        Unknown extensions fall back to ``application/octet-stream`` per HTTP convention.
        """

        content_type = GCSImageStorage._GCSImageStorage__content_type_for(  # type: ignore[attr-defined]
            filename="xmls/2026-06-04/session__1/dump.bin"
        )

        self.assertEqual(content_type, "application/octet-stream")


if __name__ == "__main__":
    unittest.main()
