from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

from fathom.base.paths import SharedPathManager
from fathom.constants.collaboration import ArtifactKind
from fathom.core.services.artifacts import ArtifactCatalog


class _CatalogFixture:
    """
    Build a path-classification-only catalog whose path manager is never invoked.
    """

    @classmethod
    def catalog(cls) -> ArtifactCatalog:
        """
        Return a catalog suitable for the pure path-classification tests.
        """

        return ArtifactCatalog(path_manager=MagicMock(spec=SharedPathManager))


class TestArtifactCatalogKind(unittest.TestCase):
    """
    ArtifactCatalog.kind returns None for scripts and an ArtifactKind for public artifacts.
    """

    def setUp(self) -> None:
        """
        Build a catalog instance with no path manager, used for path-only classification.
        """

        self.__catalog: ArtifactCatalog = _CatalogFixture.catalog()

    def test_script_text_file_returns_none(self) -> None:
        """
        A `script.txt` path is the script-content artifact and must not be classified.
        """

        self.assertIsNone(self.__catalog.kind(path=Path("/tmp/run/script.txt")))

    def test_trace_category_returns_trace_kind(self) -> None:
        """
        Files under a `traces/` category folder resolve to ArtifactKind.TRACE.
        """

        path = Path("/assets/traces/2026-05-08/com.app/workflow-1/step-0.png")
        self.assertEqual(self.__catalog.kind(path=path), ArtifactKind.TRACE)

    def test_screenshot_category_returns_screenshot_kind(self) -> None:
        """
        Files under `screenshot/` or `annotated/` resolve to ArtifactKind.SCREENSHOT.
        """

        screenshot = Path("/assets/screenshot/2026-05-08/com.app/workflow-1/step-0.png")
        annotated = Path("/assets/annotated/2026-05-08/com.app/workflow-1/step-0.png")
        self.assertEqual(self.__catalog.kind(path=screenshot), ArtifactKind.SCREENSHOT)
        self.assertEqual(self.__catalog.kind(path=annotated), ArtifactKind.SCREENSHOT)

    def test_unknown_category_falls_back_to_structured_log(self) -> None:
        """
        Any other category resolves to ArtifactKind.STRUCTURED_LOG.
        """

        path = Path("/assets/whatever/2026-05-08/com.app/workflow-1/step-0.json")
        self.assertEqual(self.__catalog.kind(path=path), ArtifactKind.STRUCTURED_LOG)


class TestArtifactCatalogMime(unittest.TestCase):
    """
    ArtifactCatalog.mime returns media types by file suffix; no script.txt special case.
    """

    def setUp(self) -> None:
        """
        Build a catalog instance for suffix-based MIME classification.
        """

        self.__catalog: ArtifactCatalog = _CatalogFixture.catalog()

    def test_script_text_file_returns_plain_text(self) -> None:
        """
        `script.txt` falls through to the `.txt` suffix and resolves to text/plain.
        """

        self.assertEqual(self.__catalog.mime(path=Path("/tmp/run/script.txt")), "text/plain")

    def test_png_returns_image_png(self) -> None:
        """
        `.png` paths resolve to image/png.
        """

        self.assertEqual(self.__catalog.mime(path=Path("a.png")), "image/png")

    def test_unknown_suffix_returns_application_json(self) -> None:
        """
        Unknown suffixes fall back to application/json.
        """

        self.assertEqual(self.__catalog.mime(path=Path("a.bin")), "application/json")


class TestArtifactCatalogRetention(unittest.TestCase):
    """
    ArtifactCatalog.retention returns no retention class for any path.
    """

    def setUp(self) -> None:
        """
        Build a catalog instance used to confirm the retention policy hook is open.
        """

        self.__catalog: ArtifactCatalog = _CatalogFixture.catalog()

    def test_script_text_file_returns_none(self) -> None:
        """
        `script.txt` no longer carries a retention class (handled via ScriptPort).
        """

        self.assertIsNone(self.__catalog.retention(path=Path("/tmp/run/script.txt")))

    def test_arbitrary_path_returns_none(self) -> None:
        """
        Arbitrary paths return None by default.
        """

        self.assertIsNone(self.__catalog.retention(path=Path("/var/data/anything.bin")))
