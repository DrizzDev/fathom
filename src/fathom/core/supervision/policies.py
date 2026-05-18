from __future__ import annotations

from typing import Optional

from fathom.constants import SWIPE_ACTIONS, ActionType
from fathom.constants.safety import UNSAFE_ACTION_KEYWORDS
from fathom.constants.screen import NO_PROGRESS_RECOVERY_THRESHOLD
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import OverlayObservation, ScreenObservation
from fathom.schemas.supervision import BlockReason
from fathom.schemas.tasks import ExecutionTask, TaskAttemptState


class TargetPolicy:
    """
    Blocks actions whose target cannot be localized safely.
    """

    def evaluate(self, *, localization: LocalizationResult) -> Optional[BlockReason]:
        """
        Return a block reason when localization is not executable.
        """

        if localization.status == LocalizationStatus.UNRESOLVED:
            return BlockReason.TARGET_UNRESOLVED

        if localization.status == LocalizationStatus.AMBIGUOUS:
            return BlockReason.TARGET_AMBIGUOUS

        return None


class RepetitionPolicy:
    """
    Blocks repeated no-progress actions before another execution.
    """

    def evaluate(self, *, count: int) -> Optional[BlockReason]:
        """
        Return a block reason when no-progress count exceeds budget.
        """

        if count >= NO_PROGRESS_RECOVERY_THRESHOLD:
            return BlockReason.REPEATED_NO_EFFECT

        return None


class KeyboardPolicy:
    """
    Blocks scroll gestures while the software keyboard is visible.
    """

    def evaluate(self, *, action: Action, observation: ScreenObservation) -> Optional[BlockReason]:
        """
        Return a block reason when keyboard state makes the action invalid.
        """

        if not observation.keyboard.visible:
            return None

        if action.action_type.value in SWIPE_ACTIONS:
            return BlockReason.KEYBOARD_OCCLUDING

        return None


class ScrollPolicy:
    """
    Blocks blind scrolling when screen evidence says scrolling is not useful.
    """

    def evaluate(
        self,
        *,
        action: Action,
        no_progress: int,
        observation: ScreenObservation,
    ) -> Optional[BlockReason]:
        """
        Return a block reason when scrolling should not execute.
        """

        if action.action_type.value not in SWIPE_ACTIONS:
            return None

        if observation.calls_to_action and no_progress > 0:
            return BlockReason.NON_SCROLLABLE_SURFACE

        if not observation.scroll and no_progress > 0:
            return BlockReason.NON_SCROLLABLE_SURFACE

        return None


class OverlayPolicy:
    """
    Blocks taps behind visible overlays.
    """

    def evaluate(
        self,
        *,
        action: Action,
        observation: ScreenObservation,
        localization: LocalizationResult,
    ) -> Optional[BlockReason]:
        """
        Return a block reason when action target is behind an overlay.
        """

        if not observation.overlays:
            return None

        if action.action_type in {ActionType.BACK, ActionType.WAIT}:
            return None

        if localization.bounds is None:
            return None

        if any(
            self.__inside_overlay_candidate(bounds=localization.bounds, overlay=overlay)
            for overlay in observation.overlays
        ):
            return None

        return BlockReason.OVERLAY_STILL_PRESENT

    def __inside_overlay_candidate(
        self,
        *,
        bounds: Bounds,
        overlay: OverlayObservation,
    ) -> bool:
        """
        Return whether bounds match one of the overlay candidates.
        """

        return any(
            self.__intersects(first=bounds, second=candidate.bounds)
            for candidate in overlay.candidates
        )

    @staticmethod
    def __intersects(*, first: Bounds, second: Bounds) -> bool:
        """
        Return whether two bounds intersect.
        """

        top = max(first.y, second.y)
        left = max(first.x, second.x)

        right = min(first.x + first.width, second.x + second.width)
        bottom = min(first.y + first.height, second.y + second.height)

        return right > left and bottom > top


class BudgetPolicy:
    """
    Blocks further actions when the active task has exhausted its attempt budget.
    """

    def evaluate(self, *, task: Optional[ExecutionTask]) -> Optional[BlockReason]:
        """
        Return a block reason when the active task has crossed its attempt cap.
        """

        if task is None:
            return None

        attempts: TaskAttemptState = task.attempts
        if attempts.limit <= 0:
            return None

        if attempts.count >= attempts.limit:
            return BlockReason.TASK_BUDGET_EXCEEDED

        return None


class SafetyPolicy:
    """
    Blocks actions whose textual context flags a potentially destructive intent.
    """

    def evaluate(self, *, action: Action) -> Optional[BlockReason]:
        """
        Return a block reason when the action's text overlaps the unsafe vocabulary.
        """

        haystack = " ".join(
            value
            for value in (
                action.target,
                action.rationale,
                action.export_target,
                action.script_target,
                action.natural_language_target,
            )
            if value
        ).lower()

        if any(keyword in haystack for keyword in UNSAFE_ACTION_KEYWORDS):
            return BlockReason.UNSAFE_ACTION

        return None
