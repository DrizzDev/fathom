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
from fathom.schemas.actions import Action
from fathom.schemas.supervision import BlockReason

if TYPE_CHECKING:
    from fathom.core.recovery.factory import RecoveryContext


class KeyboardRecovery(RecoveryStrategy):
    """
    Dismisses an occluding keyboard so the underlying surface is reachable.
    """

    @classmethod
    def build(cls, context: "RecoveryContext") -> "KeyboardRecovery":
        """
        Construct a :class:`KeyboardRecovery` from the factory context.
        """

        _ = context
        return cls()

    @property
    def name(self) -> str:
        """
        Stable identifier used in configuration and logs.
        """

        return "keyboard"

    def supports(self, *, trigger: RecoveryTrigger) -> bool:
        """
        Return whether this strategy handles the given trigger.
        """

        return trigger in (
            RecoveryTrigger.NO_PROGRESS,
            RecoveryTrigger.ACTION_BLOCKED,
            RecoveryTrigger.REQUEST_REPLAN,
        )

    async def recover(self, *, request: RecoveryRequest) -> Optional[RecoveryOutcome]:
        """
        Propose a dismissal action when the keyboard is occluding the screen.
        """

        if request.block_reason not in (BlockReason.KEYBOARD_OCCLUDING, None):
            return NoopOutcome(summary="KeyboardRecovery: block reason is not keyboard-related.")

        if request.observation is None or not request.observation.keyboard.visible:
            return NoopOutcome(summary="KeyboardRecovery: keyboard is not visible.")

        candidate = self.__first_dismiss_candidate(request=request)

        if candidate is not None:
            return TryActionOutcome(
                action=candidate,
                summary=f"KeyboardRecovery: dismissing keyboard via {candidate.target!r}.",
            )

        return TryActionOutcome(
            action=self.__hide_keyboard_action(),
            summary="KeyboardRecovery: dismissing keyboard via HIDE_KEYBOARD.",
        )

    @staticmethod
    def __first_dismiss_candidate(*, request: RecoveryRequest) -> Optional[Action]:
        """
        Return the first known keyboard-dismiss candidate as a tap action.
        """

        observation = request.observation

        if observation is None:
            return None

        for element in observation.keyboard.dismiss:
            return Action(
                confidence=0.8,
                action_type=ActionType.TAP,
                label_id=element.identifier,
                target=element.text or element.identifier,
                rationale="Keyboard recovery: dismiss via known control.",
            )

        return None

    @staticmethod
    def __hide_keyboard_action() -> Action:
        """
        Return the platform-neutral HIDE_KEYBOARD gesture used as the dismissal fallback.
        """

        return Action(
            confidence=0.8,
            target="system: hide_keyboard",
            action_type=ActionType.HIDE_KEYBOARD,
            rationale="Keyboard recovery: dismiss via HIDE_KEYBOARD platform gesture.",
        )
