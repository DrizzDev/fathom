"""
Inline defect detection from a single exploration step's runtime signals.
"""

from __future__ import annotations

from typing import List

from fathom.constants.defect import DefectSignal, DefectSource
from fathom.interfaces.defect import InlineDefectDetectorPort
from fathom.schemas.defect import Defect, DefectEvidence, StepSignals


class InlineDefectDetector(InlineDefectDetectorPort):
    """
    Flags functional defects that a single step's signals already reveal.
    """

    def inspect_step(self, *, signals: StepSignals) -> List[Defect]:
        """
        Returns the defects evidenced by one completed step.
        """

        defects: List[Defect] = []

        if signals.left_package:
            defects.append(
                self.__defect(
                    signals=signals,
                    signal=DefectSignal.LEFT_PACKAGE,
                    summary=(
                        f"Action '{signals.action_target or 'unknown'}' left the app "
                        "and BACK did not return"
                    ),
                )
            )

        if not signals.usable_capture:
            defects.append(
                self.__defect(
                    signals=signals,
                    signal=DefectSignal.BLANK_CAPTURE,
                    summary="Screen returned no usable content after the action",
                )
            )

        if signals.expects_transition and not signals.screen_changed:
            defects.append(
                self.__defect(
                    signals=signals,
                    signal=DefectSignal.DEAD_TAP,
                    summary=(
                        f"'{signals.action_target or 'control'}' was expected to change "
                        "the screen but nothing happened"
                    ),
                )
            )

        return defects

    @staticmethod
    def __defect(*, signals: StepSignals, signal: DefectSignal, summary: str) -> Defect:
        """
        Builds an inline defect anchored to the step's pre-action screen.
        """

        return Defect.from_signal(
            signal=signal,
            source=DefectSource.INLINE,
            summary=summary,
            evidence=DefectEvidence(
                screen=signals.screen,
                activity=signals.activity,
                excerpt=signals.action_target,
            ),
        )
