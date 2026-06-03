from __future__ import annotations

from typing import Optional, Tuple

from fathom.constants import ActionType
from fathom.constants.state import (
    LOOP_BACK_CONFIDENCE,
    LOOP_BACK_RATIONALE,
    LOOP_HOME_CONFIDENCE,
    LOOP_HOME_RATIONALE,
    LOOP_SCROLL_ACTION_TYPES,
    LOOP_SCROLL_CONFIDENCE,
    LOOP_SCROLL_RATIONALE,
)
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import DeviceCapability
from fathom.schemas.state import LoopDetector


class LoopActionLadder:
    """
    Ordered mechanical loop-recovery ladder.

    The full rung sequence is BACK -> SCROLL -> HOME. Rungs whose action type is not supported
    by the active device (per :class:`DeviceCapability`) are filtered out at construction so the
    ladder only ever emits actions the device can execute. SCROLL is additionally suppressed at runtime
    when the detector evidence already came from a scroll-like action, to avoid a no-op recovery on an already-scrolling stuck signal.
    """

    def __init__(self, *, device: Optional[DeviceCapability] = None) -> None:
        """
        Build the ladder with rungs filtered by the device's capabilities.
        """

        capability = device if device is not None else DeviceCapability()

        all_rungs: Tuple[Action, ...] = (
            Action(
                target="system: back",
                action_type=ActionType.BACK,
                rationale=LOOP_BACK_RATIONALE,
                confidence=LOOP_BACK_CONFIDENCE,
            ),
            Action(
                target="system: scroll",
                action_type=ActionType.SCROLL,
                rationale=LOOP_SCROLL_RATIONALE,
                confidence=LOOP_SCROLL_CONFIDENCE,
            ),
            Action(
                target="system: home",
                action_type=ActionType.HOME,
                rationale=LOOP_HOME_RATIONALE,
                confidence=LOOP_HOME_CONFIDENCE,
            ),
        )
        self.__rungs: Tuple[Action, ...] = tuple(
            rung for rung in all_rungs if capability.supports(action_type=rung.action_type)
        )

    def next(self, *, detector: LoopDetector) -> Optional[Action]:
        """
        Return the next loop-breaking action when the detector still has budget; None when passive.
        """

        if not detector.can_recover() or not self.__rungs:
            return None

        attempt = detector.record_recovery_attempt()
        return self.__pick_rung(attempt=attempt, detector=detector)

    def __pick_rung(self, *, attempt: int, detector: LoopDetector) -> Optional[Action]:
        """
        Pick the rung; passive-VALIDATE turns get no mechanical recovery so the agent re-plans on its own.
        """

        if self.__is_passive_validate(detector=detector):
            return None

        index = min(attempt - 1, len(self.__rungs) - 1)
        rung = self.__rungs[index]

        if rung.action_type is ActionType.SCROLL and self.__is_scroll_loop(detector=detector):
            return self.__advance_past_scroll(start=index)

        return rung

    def __advance_past_scroll(self, *, start: int) -> Action:
        """
        Return the next non-scroll rung after ``start`` index, or the last rung.
        """

        for candidate in self.__rungs[start + 1 :]:
            if candidate.action_type is not ActionType.SCROLL:
                return candidate

        return self.__rungs[-1]

    @staticmethod
    def __is_scroll_loop(*, detector: LoopDetector) -> bool:
        """
        Return whether the stuck evidence already came from a scroll-like action.
        """

        action_type = detector.last_action_type
        return action_type in LOOP_SCROLL_ACTION_TYPES if action_type else False

    @staticmethod
    def __is_passive_validate(*, detector: LoopDetector) -> bool:
        """
        A VALIDATE that produced no-progress is the expected outcome of a read action, not a stuck signal.
        Emitting BACK/SCROLL/HOME here can navigate away from the app, so the ladder stays silent this turn.
        """

        return detector.last_action_type == ActionType.VALIDATE.value
