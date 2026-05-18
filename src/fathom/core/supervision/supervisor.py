from __future__ import annotations

from typing import Optional

from fathom.core.runtime import RuntimeState
from fathom.core.supervision.policies import (
    BudgetPolicy,
    KeyboardPolicy,
    OverlayPolicy,
    RepetitionPolicy,
    ScrollPolicy,
    TargetPolicy,
)
from fathom.schemas.actions import Action
from fathom.schemas.localization import LocalizationResult
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.supervision import BlockReason, SupervisionVerdict, VerdictKind


class RuntimeSupervisor:
    """
    Applies deterministic pre-execution runtime policies.
    """

    # Block reasons whose verdict is informational and should NOT poison the
    # action in failure memory: the planner is allowed to re-propose the same
    # action on the next turn (e.g. once a target becomes visible).
    __NON_RECORDED_REASONS = frozenset(
        {
            BlockReason.TARGET_UNRESOLVED,
            BlockReason.TARGET_AMBIGUOUS,
        }
    )

    def __init__(
        self,
        *,
        target: TargetPolicy,
        scroll: ScrollPolicy,
        budget: BudgetPolicy,
        overlay: OverlayPolicy,
        keyboard: KeyboardPolicy,
        repetition: RepetitionPolicy,
    ) -> None:
        """
        Initialize the supervisor with the runtime-only policy set.

        Safety-vocabulary screening lives in
        :class:`fathom.core.safety.classifier.IntentSafetyClassifier` and
        is consulted before workflow start, not on every step. The
        per-step ``SafetyPolicy`` is intentionally not wired here — its
        substring scan over the LLM's rationale produced false positives
        on every ``swipe`` (``"wipe" in "swipe"``).
        """

        self.__target = target
        self.__scroll = scroll
        self.__budget = budget
        self.__overlay = overlay
        self.__keyboard = keyboard
        self.__repetition = repetition

    @classmethod
    def create(cls) -> "RuntimeSupervisor":
        """
        Create a supervisor with default runtime policies.
        """

        return cls(
            target=TargetPolicy(),
            scroll=ScrollPolicy(),
            budget=BudgetPolicy(),
            overlay=OverlayPolicy(),
            keyboard=KeyboardPolicy(),
            repetition=RepetitionPolicy(),
        )

    def supervise(
        self,
        *,
        action: Action,
        runtime: RuntimeState,
        observation: ScreenObservation,
        localization: LocalizationResult,
    ) -> SupervisionVerdict:
        """
        Return whether an action may execute.
        """

        reason = self.__first_block_reason(
            action=action,
            runtime=runtime,
            observation=observation,
            localization=localization,
        )
        if reason is not None:
            if reason not in self.__NON_RECORDED_REASONS:
                runtime.failures.record(
                    action=action,
                    reason=reason,
                    detail=f"Runtime supervision blocked action: {reason.value}",
                )
            return SupervisionVerdict(
                action=None,
                reason=reason,
                kind=VerdictKind.BLOCK,
                localization=localization,
                message=f"Runtime supervision blocked action: {reason.value}",
            )

        return SupervisionVerdict(
            reason=None,
            action=action,
            kind=VerdictKind.ALLOW,
            localization=localization,
            message="Action approved for execution.",
        )

    def __first_block_reason(
        self,
        *,
        action: Action,
        runtime: RuntimeState,
        observation: ScreenObservation,
        localization: LocalizationResult,
    ) -> Optional[BlockReason]:
        """
        Return the first applicable block reason.
        """

        no_progress = runtime.effects.consecutive_no_progress()

        for reason in (
            self.__target.evaluate(localization=localization),
            self.__budget.evaluate(task=runtime.tasks.active()),
            self.__keyboard.evaluate(action=action, observation=observation),
            self.__scroll.evaluate(
                action=action,
                observation=observation,
                no_progress=no_progress,
            ),
            self.__overlay.evaluate(
                action=action,
                observation=observation,
                localization=localization,
            ),
            self.__repetition.evaluate(count=no_progress),
        ):
            if reason is not None:
                return reason

        return None
