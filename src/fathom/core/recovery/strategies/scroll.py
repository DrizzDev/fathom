from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from fathom.constants import ActionType
from fathom.core.recovery.strategy import RecoveryStrategy
from fathom.core.recovery.types import (
    NoopOutcome,
    RecoveryOutcome,
    RecoveryRequest,
    RecoveryTrigger,
    TryActionOutcome,
)
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.observation import PerceivedElement
from fathom.schemas.supervision import BlockReason

SCROLL_ACTION_TYPES: Tuple[ActionType, ...] = (
    ActionType.SCROLL,
    ActionType.SWIPE,
    ActionType.SWIPE_UP,
    ActionType.SWIPE_DOWN,
    ActionType.SWIPE_LEFT,
    ActionType.SWIPE_RIGHT,
)

if TYPE_CHECKING:
    from fathom.core.recovery.factory import RecoveryContext


class ScrollBoundaryRecovery(RecoveryStrategy):
    """
    Surfaces a visible terminal control instead of continuing an ineffective scroll.
    """

    @classmethod
    def build(cls, context: "RecoveryContext") -> "ScrollBoundaryRecovery":
        """
        Construct a :class:`ScrollBoundaryRecovery` from the factory context.
        """

        _ = context
        return cls()

    @property
    def name(self) -> str:
        """
        Stable identifier used in configuration and logs.
        """

        return "scroll"

    def supports(self, *, trigger: RecoveryTrigger) -> bool:
        """
        Return whether this strategy handles the given trigger.
        """

        return trigger in (
            RecoveryTrigger.NO_PROGRESS,
            RecoveryTrigger.LOOP_DETECTED,
            RecoveryTrigger.ACTION_BLOCKED,
            RecoveryTrigger.REQUEST_REPLAN,
        )

    async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
        """
        Propose tapping a visible call-to-action when scrolling is ineffective.
        """

        if request.block_reason not in (
            None,
            BlockReason.REPEATED_NO_EFFECT,
            BlockReason.NON_SCROLLABLE_SURFACE,
        ):
            return NoopOutcome(
                summary="ScrollBoundaryRecovery: block reason is not scroll-related."
            )

        if request.observation is None:
            return NoopOutcome(summary="ScrollBoundaryRecovery: no screen observation available.")

        cta = self.__first_unused_cta(request=request)
        if cta is None:
            return NoopOutcome(
                summary="ScrollBoundaryRecovery: no visible call-to-action to surface."
            )

        return TryActionOutcome(
            action=self.__build_action(element=cta),
            summary=f"ScrollBoundaryRecovery: surfacing CTA {cta.identifier!r} instead of scroll.",
        )

    @staticmethod
    def __first_unused_cta(*, request: RecoveryRequest) -> Optional[PerceivedElement]:
        """
        Return the first visible CTA that has not been used recently.
        """

        observation = request.observation

        if observation is None:
            return None

        recent = {entry.lower() for entry in request.recent_actions}

        for element in observation.calls_to_action:
            label = element.text or element.identifier

            if label.lower() not in recent:
                return element

        return None

    @staticmethod
    def __build_action(*, element: PerceivedElement) -> Action:
        """
        Build a tap action targeting the supplied call-to-action element.
        """

        bounds = element.bounds

        return Action(
            confidence=0.75,
            action_type=ActionType.TAP,
            label_id=element.identifier,
            target=element.text or element.identifier,
            rationale="Scroll boundary recovery: tap visible terminal control.",
            bounds=Bounds(
                x=bounds.x,
                y=bounds.y,
                width=bounds.width,
                height=bounds.height,
                coordinate_system="pixel",
            ),
        )
