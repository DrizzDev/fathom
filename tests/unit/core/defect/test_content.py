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

    async def test_skeleton_placeholder_description_is_not_flagged(self) -> None:
        """
        'placeholder'/'skeleton' in a screen description is not a content defect.

        These words legitimately describe loading skeletons; flagging them produced
        false positives, so only strong markers like lorem-ipsum remain.
        """

        snapshot = ScreenSnapshot(
            screen="hash-1",
            texts=["Swiggy hub showing placeholder loading skeletons at the bottom"],
        )

        self.assertEqual(await self.__detector.inspect_screen(snapshot=snapshot), [])

    async def test_word_boundary_avoids_false_positive(self) -> None:
        """
        A marker embedded in a larger word is not flagged.
        """

        snapshot = ScreenSnapshot(screen="hash-1", texts=["Open the loremised mockups"])

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
