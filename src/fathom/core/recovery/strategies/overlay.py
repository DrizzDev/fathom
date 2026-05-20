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
from fathom.schemas.observation import OverlayObservation, PerceivedElement
from fathom.schemas.supervision import BlockReason

if TYPE_CHECKING:
    from fathom.core.recovery.factory import RecoveryContext


class OverlayRecovery(RecoveryStrategy):
    """
    Proposes the next dismiss candidate when a blocking overlay persists.
    """

    @classmethod
    def build(cls, context: "RecoveryContext") -> "OverlayRecovery":
        """
        Construct an :class:`OverlayRecovery` from the factory context.
        """

        _ = context
        return cls()

    @property
    def name(self) -> str:
        """
        Stable identifier used in configuration and logs.
        """

        return "overlay"

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
        Pick the next overlay dismiss candidate or defer to the next strategy.
        """

        if request.block_reason not in (BlockReason.OVERLAY_STILL_PRESENT, None):
            return NoopOutcome(summary="OverlayRecovery: block reason is not overlay-related.")

        if request.observation is None:
            return NoopOutcome(summary="OverlayRecovery: no screen observation available.")

        overlay = self.__first_visible_overlay(request=request)

        if overlay is None:
            return NoopOutcome(summary="OverlayRecovery: no visible overlay detected.")

        candidate = self.__first_unused_dismiss(overlay=overlay, request=request)

        if candidate is None:
            return NoopOutcome(summary="OverlayRecovery: no unused dismiss candidate available.")

        return TryActionOutcome(
            action=self.__build_action(candidate=candidate),
            summary=f"OverlayRecovery: dismissing overlay via {candidate.identifier!r}.",
        )

    @staticmethod
    def __first_visible_overlay(*, request: RecoveryRequest) -> Optional[OverlayObservation]:
        """
        Return the first visible overlay in the current observation.
        """

        observation = request.observation

        if observation is None:
            return None

        for overlay in observation.overlays:
            if overlay.visible and overlay.candidates:
                return overlay

        return None

    @staticmethod
    def __first_unused_dismiss(
        *,
        request: RecoveryRequest,
        overlay: OverlayObservation,
    ) -> Optional[PerceivedElement]:
        """
        Return the first dismiss candidate that has not been tried recently.
        """

        recent = {entry.lower() for entry in request.recent_actions}

        for candidate in overlay.candidates:
            label = candidate.text or candidate.identifier

            if label.lower() not in recent:
                return candidate

        return None

    @staticmethod
    def __build_action(*, candidate: PerceivedElement) -> Action:
        """
        Build a tap action targeting the supplied perceived element.
        """

        bounds = candidate.bounds

        return Action(
            confidence=0.8,
            action_type=ActionType.TAP,
            label_id=candidate.identifier,
            target=candidate.text or candidate.identifier,
            rationale="Overlay recovery: tap next dismiss candidate.",
            bounds=Bounds(
                x=bounds.x,
                y=bounds.y,
                width=bounds.width,
                height=bounds.height,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
        )
