from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fathom.constants import ActionType
from fathom.core.recovery.strategy import RecoveryStrategy
from fathom.core.recovery.types import (
    NoopOutcome,
    RecoveryOutcome,
    RecoveryRequest,
    RecoveryTrigger,
    TryActionOutcome,
)
from fathom.schemas.actions import Action, Bounds, CoordinateSystem
from fathom.schemas.localization import LocalizationCandidate
from fathom.schemas.observation import PerceivedElement
from fathom.schemas.supervision import BlockReason

if TYPE_CHECKING:
    from fathom.core.recovery.factory import RecoveryContext


class AlternativeTargetRecovery(RecoveryStrategy):
    """
    Retries with the next localization candidate when the original target is unusable.
    """

    @classmethod
    def build(cls, context: "RecoveryContext") -> "AlternativeTargetRecovery":
        """
        Construct an :class:`AlternativeTargetRecovery` from the factory context.
        """

        _ = context
        return cls()

    @property
    def name(self) -> str:
        """
        Stable identifier used in configuration and logs.
        """

        return "alternative"

    def supports(self, *, trigger: RecoveryTrigger) -> bool:
        """
        Return whether this strategy handles the given trigger.
        """

        return trigger in (
            RecoveryTrigger.NO_PROGRESS,
            RecoveryTrigger.ACTION_BLOCKED,
            RecoveryTrigger.REQUEST_REPLAN,
            RecoveryTrigger.TARGET_UNRESOLVED,
        )

    async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
        """
        Propose the next localization candidate as a retry action.
        """

        if request.block_reason not in (
            BlockReason.TARGET_AMBIGUOUS,
            BlockReason.TARGET_UNRESOLVED,
            BlockReason.REPEATED_NO_EFFECT,
            None,
        ):
            return NoopOutcome(
                summary="AlternativeTargetRecovery: block reason is not target-related."
            )

        candidate = self.__first_unused_candidate(request=request)
        if candidate is None:
            return NoopOutcome(
                summary="AlternativeTargetRecovery: no unused localization candidate available."
            )

        return TryActionOutcome(
            action=self.__build_action(candidate=candidate),
            summary=f"AlternativeTargetRecovery: retrying with candidate {candidate.reason!r}.",
        )

    @staticmethod
    def __first_unused_candidate(
        *,
        request: RecoveryRequest,
    ) -> Optional[LocalizationCandidate]:
        """
        Return the first localization candidate not yet attempted.
        """

        recent = {entry.lower() for entry in request.recent_actions}

        for candidate in request.candidates:
            label = AlternativeTargetRecovery.__label_for(candidate=candidate)

            if label.lower() not in recent:
                return candidate

        return None

    @staticmethod
    def __label_for(*, candidate: LocalizationCandidate) -> str:
        """
        Return the human-facing label used for trace comparison.
        """

        element = candidate.element

        if element is not None and element.text:
            return element.text

        if element is not None:
            return element.identifier

        return candidate.reason

    @staticmethod
    def __build_action(*, candidate: LocalizationCandidate) -> Action:
        """
        Build a tap action targeting the supplied localization candidate.
        """

        element = candidate.element
        bounds = AlternativeTargetRecovery.__bounds_for(element=element)

        return Action(
            bounds=bounds,
            confidence=candidate.score,
            action_type=ActionType.TAP,
            rationale=f"Alternative target recovery: {candidate.reason}",
            label_id=element.identifier if element is not None else None,
            target=AlternativeTargetRecovery.__label_for(candidate=candidate),
        )

    @staticmethod
    def __bounds_for(*, element: Optional[PerceivedElement]) -> Optional[Bounds]:
        """
        Return executable bounds for the supplied element when available.
        """

        if element is None:
            return None

        source = element.bounds

        return Bounds(
            x=source.x,
            y=source.y,
            width=source.width,
            height=source.height,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )
