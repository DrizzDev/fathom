from __future__ import annotations

import unittest

from fathom.constants.defect import (
    DefectKind,
    DefectSeverity,
    DefectSignal,
    DefectSource,
    DefectVerification,
)


class DefectSeverityTest(unittest.TestCase):
    """
    Pins the severity ordering the bug report sorts on.
    """

    def test_rank_orders_blocker_before_info(self) -> None:
        """
        Ranks ascend from BLOCKER (most severe) to INFO (least).
        """

        ordered = sorted(DefectSeverity, key=lambda severity: severity.rank)
        self.assertEqual(
            ordered,
            [
                DefectSeverity.BLOCKER,
                DefectSeverity.MAJOR,
                DefectSeverity.MINOR,
                DefectSeverity.INFO,
            ],
        )


class DefectSignalTest(unittest.TestCase):
    """
    Guards the signal taxonomy and its mapping to kinds and severities.
    """

    def test_every_signal_maps_to_a_kind(self) -> None:
        """
        Each signal resolves a kind; an unmapped signal raises instead of passing silently.
        """

        for signal in DefectSignal:
            self.assertIsInstance(signal.kind, DefectKind)

    def test_every_signal_has_a_default_severity(self) -> None:
        """
        Each signal resolves a default severity.
        """

        for signal in DefectSignal:
            self.assertIsInstance(signal.default_severity, DefectSeverity)

    def test_crash_is_a_blocking_functional_defect(self) -> None:
        """
        A crash is the canonical blocker-severity functional defect.
        """

        self.assertEqual(DefectSignal.CRASH.kind, DefectKind.FUNCTIONAL)
        self.assertEqual(DefectSignal.CRASH.default_severity, DefectSeverity.BLOCKER)

    def test_lorem_ipsum_is_a_content_defect(self) -> None:
        """
        Lorem-ipsum copy is categorised as a content defect.
        """

        self.assertEqual(DefectSignal.LOREM_IPSUM.kind, DefectKind.CONTENT)

    def test_overlap_clipping_is_a_ui_defect(self) -> None:
        """
        Overlap and clipping are visual defects.
        """

        self.assertEqual(DefectSignal.OVERLAP_CLIPPING.kind, DefectKind.UI)


class DefectSourceTest(unittest.TestCase):
    """
    Pins the wire values distinguishing live from post-run detection.
    """

    def test_source_values(self) -> None:
        """
        Sources serialize to stable lower-case tokens.
        """

        self.assertEqual(DefectSource.INLINE.value, "inline")
        self.assertEqual(DefectSource.POST_RUN.value, "post_run")


class DefectVerificationTest(unittest.TestCase):
    """
    Pins the verification states that gate whether a defect leads the report.
    """

    def test_values(self) -> None:
        """
        Verification states serialize to stable lower-case tokens.
        """

        self.assertEqual(DefectVerification.CONFIRMED.value, "confirmed")
        self.assertEqual(DefectVerification.NEEDS_REVIEW.value, "needs_review")


if __name__ == "__main__":
    unittest.main()
