from __future__ import annotations

import unittest
from typing import Any

from fathom.constants.defect import DefectSignal, DefectSource
from fathom.core.defect.inline import InlineDefectDetector
from fathom.schemas.defect import StepSignals


class InlineDefectDetectorTest(unittest.TestCase):
    """
    Verifies inline detection from a step's runtime signals.
    """

    def setUp(self) -> None:
        """
        Builds the detector under test.
        """

        self.__detector = InlineDefectDetector()

    @staticmethod
    def __signals(**overrides: Any) -> StepSignals:
        """
        Builds a healthy step's signals, overriding only the fields under test.
        """

        base = {
            "screen": "hash-1",
            "activity": "com.app/.Home",
            "action_target": "Buy",
            "expects_transition": False,
            "screen_changed": True,
            "left_package": False,
            "usable_capture": True,
        }
        base.update(overrides)
        return StepSignals(**base)

    def test_dead_tap_when_transition_expected_but_screen_unchanged(self) -> None:
        """
        A predicted transition that does not happen is a dead tap.
        """

        defects = self.__detector.inspect_step(
            signals=self.__signals(expects_transition=True, screen_changed=False)
        )
        self.assertEqual([defect.signal for defect in defects], [DefectSignal.DEAD_TAP])

    def test_no_dead_tap_when_no_transition_expected(self) -> None:
        """
        An unchanged screen is not a defect when no transition was predicted.
        """

        defects = self.__detector.inspect_step(
            signals=self.__signals(expects_transition=False, screen_changed=False)
        )
        self.assertEqual(defects, [])

    def test_no_dead_tap_when_screen_changed(self) -> None:
        """
        A predicted transition that happens is not a defect.
        """

        defects = self.__detector.inspect_step(
            signals=self.__signals(expects_transition=True, screen_changed=True)
        )
        self.assertEqual(defects, [])

    def test_left_package_flagged(self) -> None:
        """
        An unrecoverable package exit is a functional defect.
        """

        defects = self.__detector.inspect_step(signals=self.__signals(left_package=True))
        self.assertIn(DefectSignal.LEFT_PACKAGE, [defect.signal for defect in defects])

    def test_blank_capture_flagged(self) -> None:
        """
        A post-action capture with no usable screen is a defect.
        """

        defects = self.__detector.inspect_step(signals=self.__signals(usable_capture=False))
        self.assertIn(DefectSignal.BLANK_CAPTURE, [defect.signal for defect in defects])

    def test_evidence_is_anchored_to_the_pre_action_screen(self) -> None:
        """
        The defect carries the screen, activity, and inline source for triage.
        """

        (defect,) = self.__detector.inspect_step(
            signals=self.__signals(expects_transition=True, screen_changed=False)
        )
        self.assertEqual(defect.evidence.screen, "hash-1")
        self.assertEqual(defect.evidence.activity, "com.app/.Home")
        self.assertEqual(defect.source, DefectSource.INLINE)


if __name__ == "__main__":
    unittest.main()
