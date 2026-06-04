from __future__ import annotations

import unittest

from fathom.infrastructure.storage.cloud import GCSImageStorage


class GCSExtensionFallbackTest(unittest.TestCase):
    """
    Pins the category-aware extension fallback used when the caller did not supply ``meta.filename``.
    """

    def test_xmls_category_uses_xml_extension(self) -> None:
        """
        Hierarchy XML uploads with no caller-supplied filename land as ``.xml`` in GCS.
        """

        filename = GCSImageStorage._GCSImageStorage__fallback_filename(  # type: ignore[attr-defined]
            category="xmls", activity="com.app", package="com.app"
        )

        self.assertTrue(filename.endswith("__com.app__com.app.xml"))

    def test_history_category_uses_txt_extension(self) -> None:
        """
        History uploads fall back to ``.txt`` so a downloaded file opens cleanly.
        """

        filename = GCSImageStorage._GCSImageStorage__fallback_filename(  # type: ignore[attr-defined]
            category="history", activity="com.app", package="com.app"
        )

        self.assertTrue(filename.endswith(".txt"))

    def test_other_categories_default_to_png(self) -> None:
        """
        Image-bearing categories (screenshot, annotated, traces) keep the ``.png`` default.
        """

        for category in ("screenshot", "annotated", "traces"):
            with self.subTest(category=category):
                filename = GCSImageStorage._GCSImageStorage__fallback_filename(  # type: ignore[attr-defined]
                    category=category, activity="com.app", package="com.app"
                )

                self.assertTrue(filename.endswith(".png"))

    def test_filename_embeds_package_for_multi_package_session_safety(self) -> None:
        """
        Fallback filename embeds the package so multi-package sessions cannot collide on the flat path.
        """

        filename = GCSImageStorage._GCSImageStorage__fallback_filename(  # type: ignore[attr-defined]
            category="xmls", activity="com.app.activity", package="com.app"
        )

        self.assertIn("__com.app__", filename)


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
