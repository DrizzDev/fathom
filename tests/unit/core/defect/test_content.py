from __future__ import annotations

import unittest

from fathom.constants.defect import DefectKind, DefectSignal, DefectSource
from fathom.core.defect.content import ContentDefectDetector
from fathom.schemas.defect import ScreenSnapshot


class ContentDefectDetectorTest(unittest.IsolatedAsyncioTestCase):
    """
    Verifies placeholder-copy detection over a screen's text.
    """

    def setUp(self) -> None:
        """
        Builds the detector under test.
        """

        self.__detector = ContentDefectDetector()

    async def test_flags_lorem_ipsum(self) -> None:
        """
        Lorem-ipsum copy is reported as a content defect on the screen.
        """

        snapshot = ScreenSnapshot(
            screen="hash-1",
            activity="com.app/.Home",
            texts=["Welcome", "Lorem ipsum dolor sit amet"],
        )

        (defect,) = await self.__detector.inspect_screen(snapshot=snapshot)

        self.assertEqual(defect.signal, DefectSignal.LOREM_IPSUM)
        self.assertEqual(defect.kind, DefectKind.CONTENT)
        self.assertEqual(defect.source, DefectSource.POST_RUN)
        self.assertEqual(defect.evidence.screen, "hash-1")

    async def test_flags_todo_marker(self) -> None:
        """
        A bare TODO marker in the copy is flagged.
        """

        snapshot = ScreenSnapshot(screen="hash-1", texts=["TODO: wire up checkout"])

        signals = {
            defect.signal for defect in await self.__detector.inspect_screen(snapshot=snapshot)
        }

        self.assertIn(DefectSignal.TODO_TEXT, signals)

    async def test_word_boundary_avoids_false_positive(self) -> None:
        """
        A substring inside a real word (todoist) is not a TODO defect.
        """

        snapshot = ScreenSnapshot(screen="hash-1", texts=["Open Todoist to plan"])

        self.assertEqual(await self.__detector.inspect_screen(snapshot=snapshot), [])

    async def test_one_defect_per_signal(self) -> None:
        """
        Repeated markers of the same signal collapse to a single defect.
        """

        snapshot = ScreenSnapshot(
            screen="hash-1",
            texts=["Lorem ipsum here", "and lorem ipsum there"],
        )

        defects = await self.__detector.inspect_screen(snapshot=snapshot)

        self.assertEqual([defect.signal for defect in defects], [DefectSignal.LOREM_IPSUM])

    async def test_empty_text_yields_nothing(self) -> None:
        """
        A screen with no text produces no content defects.
        """

        snapshot = ScreenSnapshot(screen="hash-1", texts=[])

        self.assertEqual(await self.__detector.inspect_screen(snapshot=snapshot), [])


if __name__ == "__main__":
    unittest.main()
