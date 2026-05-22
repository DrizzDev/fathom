from __future__ import annotations

from typing import Optional

from fathom.constants import SWIPE_ACTIONS, ActionExecutionKind, ActionType
from fathom.constants.screen import (
    ACTION_EFFECT_CONTENT_DIFF_RATIO_THRESHOLD,
    ACTION_EFFECT_SCROLL_DISTANCE_THRESHOLD_PX,
    SCROLL_IDENTICAL_FRAME_SSIM_THRESHOLD,
)
from fathom.schemas.actions import Action
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus
from fathom.schemas.screens import ScreenDiff
from fathom.schemas.scroll import ScrollOutcome


class OutcomeClassifier:
    """
    Classifies post-action UI effects using action-aware rules.
    """

    def classify(
        self,
        *,
        success: bool,
        action: Action,
        before: ScreenObservation,
        diff: Optional[ScreenDiff],
        after: Optional[ScreenObservation],
        scroll_outcome: Optional[ScrollOutcome] = None,
    ) -> ActionOutcome:
        """
        Classify one attempted action outcome.
        """

        if action.execution_kind is ActionExecutionKind.CONTROL:
            return self.__outcome(
                diff=diff,
                after=after,
                before=before,
                action=action,
                status=OutcomeStatus.UNKNOWN,
                reason="Control-flow action completed outside device effect classification.",
            )

        if after is None or diff is None:
            if not success:
                return self.__outcome(
                    diff=diff,
                    after=after,
                    before=before,
                    action=action,
                    status=OutcomeStatus.BLOCKED,
                    reason="Device command failed and no post-action observation was available.",
                )
            return self.__outcome(
                diff=diff,
                after=after,
                before=before,
                action=action,
                status=OutcomeStatus.UNKNOWN,
                reason="Post-action observation was unavailable.",
            )

        outcome = self.__classify_device_action(
            action=action,
            before=before,
            after=after,
            diff=diff,
            scroll_outcome=scroll_outcome,
        )

        if success:
            return outcome

        return self.__reconcile_failed_execution(
            action=action,
            before=before,
            after=after,
            diff=diff,
            outcome=outcome,
        )

    def __classify_device_action(
        self,
        *,
        action: Action,
        before: ScreenObservation,
        after: ScreenObservation,
        diff: ScreenDiff,
        scroll_outcome: Optional[ScrollOutcome],
    ) -> ActionOutcome:
        """
        Classify one device action using post-action evidence.
        """

        if action.action_type.value in SWIPE_ACTIONS:
            return self.__classify_scroll(
                diff=diff,
                after=after,
                action=action,
                before=before,
                scroll_outcome=scroll_outcome,
            )

        if action.action_type == ActionType.TYPE:
            return self.__classify_type(
                diff=diff,
                after=after,
                action=action,
                before=before,
            )

        if action.action_type == ActionType.TAP:
            return self.__classify_tap(
                diff=diff,
                after=after,
                action=action,
                before=before,
            )

        if action.action_type in {ActionType.BACK, ActionType.HOME, ActionType.WAIT}:
            return self.__classify_generic(
                diff=diff,
                after=after,
                action=action,
                before=before,
            )

        return self.__classify_generic(
            diff=diff,
            after=after,
            action=action,
            before=before,
        )

    def __reconcile_failed_execution(
        self,
        *,
        action: Action,
        before: ScreenObservation,
        after: ScreenObservation,
        diff: ScreenDiff,
        outcome: ActionOutcome,
    ) -> ActionOutcome:
        """
        Reconcile raw device failure with post-action UI evidence.
        """

        if outcome.status is OutcomeStatus.EFFECTIVE:
            return self.__outcome(
                diff=diff,
                after=after,
                before=before,
                action=action,
                status=OutcomeStatus.EFFECTIVE,
                reason=(
                    "Device command reported failure, but post-action evidence showed "
                    f"a visible UI effect. {outcome.reason}"
                ),
            )

        if outcome.status is OutcomeStatus.NO_EFFECT:
            return self.__outcome(
                diff=diff,
                after=after,
                before=before,
                action=action,
                status=OutcomeStatus.BLOCKED,
                reason="Device command failed and no visible UI effect was detected.",
            )

        return self.__outcome(
            diff=diff,
            after=after,
            before=before,
            action=action,
            status=OutcomeStatus.UNKNOWN,
            reason=(
                "Device command reported failure and post-action evidence was inconclusive. "
                f"{outcome.reason}"
            ),
        )

    def __classify_tap(
        self,
        *,
        action: Action,
        diff: ScreenDiff,
        after: ScreenObservation,
        before: ScreenObservation,
    ) -> ActionOutcome:
        """
        Classify a tap action outcome.
        """

        if self.__overlay_count(before=before) != self.__overlay_count(before=after):
            return self.__effective(
                diff=diff,
                after=after,
                action=action,
                before=before,
                reason="Tap changed overlay state.",
            )

        return self.__from_diff(action=action, before=before, after=after, diff=diff)

    def __classify_type(
        self,
        *,
        action: Action,
        diff: ScreenDiff,
        after: ScreenObservation,
        before: ScreenObservation,
    ) -> ActionOutcome:
        """
        Classify a text input action outcome.
        """

        if diff.action_had_effect:
            return self.__effective(
                diff=diff,
                after=after,
                action=action,
                before=before,
                reason="Typing changed the visible UI state.",
            )

        return self.__no_effect(
            diff=diff,
            after=after,
            action=action,
            before=before,
            reason="Typing command succeeded but no visible UI effect was detected.",
        )

    def __classify_scroll(
        self,
        *,
        action: Action,
        diff: ScreenDiff,
        after: ScreenObservation,
        before: ScreenObservation,
        scroll_outcome: Optional[ScrollOutcome],
    ) -> ActionOutcome:
        """
        Classify a scroll or swipe action outcome.
        """

        if scroll_outcome is not None:
            verdict = scroll_outcome.final.kind
            diagnostic = scroll_outcome.final.detail or verdict.value
            base_outcome = self.__from_diff(action=action, before=before, after=after, diff=diff)
            if base_outcome.status is OutcomeStatus.EFFECTIVE:
                return base_outcome.model_copy(
                    update={
                        "reason": (
                            "Scroll produced a visible viewport change. "
                            f"Scroll diagnostic: {diagnostic}."
                        )
                    }
                )
            return base_outcome.model_copy(
                update={"reason": f"Scroll diagnostic: {diagnostic}. {base_outcome.reason}"}
            )

        if before.keyboard.visible:
            return self.__no_effect(
                diff=diff,
                after=after,
                action=action,
                before=before,
                reason="Scroll was attempted while the keyboard was visible.",
            )

        if diff.is_genuinely_different_state:
            return self.__effective(
                diff=diff,
                after=after,
                action=action,
                before=before,
                reason="Scroll produced a meaningful viewport change.",
            )

        return self.__no_effect(
            diff=diff,
            after=after,
            action=action,
            before=before,
            reason="Scroll command succeeded but no meaningful viewport change was detected.",
        )

    def __classify_generic(
        self,
        *,
        action: Action,
        diff: ScreenDiff,
        after: ScreenObservation,
        before: ScreenObservation,
    ) -> ActionOutcome:
        """
        Classify a generic action outcome.
        """

        return self.__from_diff(action=action, before=before, after=after, diff=diff)

    def __from_diff(
        self,
        *,
        action: Action,
        diff: ScreenDiff,
        after: ScreenObservation,
        before: ScreenObservation,
    ) -> ActionOutcome:
        """
        Classify an outcome using screen diff evidence.

        When :attr:`ScreenDiff.action_had_effect` is True only because
        of a structural hash signal (xml / interaction / activity) while
        every visual signal reports perfect identity (phash distance 0,
        SSIM at the identical-frame threshold, no pixel diff, no changed
        regions, no scroll displacement), the diff is treated as a
        no-op. The structural hashes flip on benign drift like the
        status-bar clock label or animation frame counters, and the
        agent must not be told an action "worked" when nothing visible
        on screen changed — otherwise the loop detector cannot escalate
        recovery.
        """

        if diff.action_had_effect and not self.__visually_identical(diff=diff):
            return self.__effective(
                diff=diff,
                after=after,
                action=action,
                before=before,
                reason="Action produced a visible UI effect.",
            )

        if diff.action_had_effect:
            return self.__no_effect(
                diff=diff,
                after=after,
                action=action,
                before=before,
                reason=(
                    "Structural hash signal indicated a change but every visual signal "
                    "showed perfect frame identity — treating as no-op."
                ),
            )

        return self.__no_effect(
            diff=diff,
            after=after,
            action=action,
            before=before,
            reason="Action command succeeded but no visible UI effect was detected.",
        )

    @staticmethod
    def __visually_identical(*, diff: ScreenDiff) -> bool:
        """
        Whether every visual evidence channel reports the two captures
        as effectively identical, regardless of structural hashes.
        """

        if diff.phash_distance != 0:
            return False
        if diff.ssim_score is None or diff.ssim_score < SCROLL_IDENTICAL_FRAME_SSIM_THRESHOLD:
            return False
        if (
            diff.content_pixel_diff_ratio is not None
            and diff.content_pixel_diff_ratio > ACTION_EFFECT_CONTENT_DIFF_RATIO_THRESHOLD
        ):
            return False
        if diff.changed_regions:
            return False
        return not (
            diff.scroll_translation is not None
            and (
                abs(diff.scroll_translation.dx) > ACTION_EFFECT_SCROLL_DISTANCE_THRESHOLD_PX
                or abs(diff.scroll_translation.dy) > ACTION_EFFECT_SCROLL_DISTANCE_THRESHOLD_PX
            )
        )

    def __effective(
        self,
        *,
        reason: str,
        action: Action,
        diff: ScreenDiff,
        after: ScreenObservation,
        before: ScreenObservation,
    ) -> ActionOutcome:
        """
        Build an effective outcome.
        """

        return self.__outcome(
            diff=diff,
            after=after,
            reason=reason,
            action=action,
            before=before,
            status=OutcomeStatus.EFFECTIVE,
        )

    def __no_effect(
        self,
        *,
        reason: str,
        action: Action,
        diff: ScreenDiff,
        after: ScreenObservation,
        before: ScreenObservation,
    ) -> ActionOutcome:
        """
        Build a no-effect outcome.
        """

        return self.__outcome(
            diff=diff,
            after=after,
            reason=reason,
            action=action,
            before=before,
            status=OutcomeStatus.NO_EFFECT,
        )

    @staticmethod
    def __outcome(
        *,
        reason: str,
        action: Action,
        status: OutcomeStatus,
        before: ScreenObservation,
        diff: Optional[ScreenDiff],
        after: Optional[ScreenObservation],
    ) -> ActionOutcome:
        """
        Build an action outcome.
        """

        return ActionOutcome(
            diff=diff,
            after=after,
            before=before,
            status=status,
            action=action,
            reason=reason,
        )

    @staticmethod
    def __overlay_count(*, before: ScreenObservation) -> int:
        """
        Return visible overlay count.
        """

        return len(before.overlays)
