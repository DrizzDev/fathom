from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import ActionType


class ActionKind(StrEnum):
    """
    Functional categorization of an action, derived from its :class:`ActionType`.

    Used to interpret loop-detector turns and to annotate past-action entries
    in the vision prompt so the model can distinguish "screen did not change
    after a validate" (expected) from "screen did not change after a tap" (a real signal of being stuck).
    """

    INPUT = "input"
    UNKNOWN = "unknown"
    TERMINAL = "terminal"
    NAVIGATION = "navigation"
    VALIDATION = "validation"
    ESCALATION = "escalation"
    OBSERVATION = "observation"


ACTION_KIND_BY_TYPE: Mapping[ActionType, ActionKind] = MappingProxyType(
    {
        ActionType.TYPE: ActionKind.INPUT,
        ActionType.TEXT: ActionKind.INPUT,
        ActionType.UNKNOWN: ActionKind.UNKNOWN,
        ActionType.TAP: ActionKind.NAVIGATION,
        ActionType.BACK: ActionKind.NAVIGATION,
        ActionType.HOME: ActionKind.NAVIGATION,
        ActionType.SWIPE: ActionKind.NAVIGATION,
        ActionType.SCROLL: ActionKind.NAVIGATION,
        ActionType.SWIPE_UP: ActionKind.NAVIGATION,
        ActionType.SWIPE_DOWN: ActionKind.NAVIGATION,
        ActionType.SWIPE_LEFT: ActionKind.NAVIGATION,
        ActionType.LONG_PRESS: ActionKind.NAVIGATION,
        ActionType.SWIPE_RIGHT: ActionKind.NAVIGATION,
        ActionType.HIDE_KEYBOARD: ActionKind.NAVIGATION,
        ActionType.COMPLETE: ActionKind.TERMINAL,
        ActionType.VALIDATE: ActionKind.VALIDATION,
        ActionType.ASK_USER: ActionKind.ESCALATION,
        ActionType.WAIT: ActionKind.OBSERVATION,
        ActionType.STORE: ActionKind.OBSERVATION,
        ActionType.INFER: ActionKind.OBSERVATION,
        ActionType.SAVE_MEMORY: ActionKind.OBSERVATION,
        ActionType.RETRIEVE_MEMORY: ActionKind.OBSERVATION,
    }
)


class ActionKindResolver:
    """
    Resolves raw and typed actions into functional action categories.
    """

    @staticmethod
    def resolve(*, action_type: ActionType) -> ActionKind:
        """
        Resolve a concrete action type to its functional kind.
        """

        return ACTION_KIND_BY_TYPE.get(action_type, ActionKind.UNKNOWN)

    @staticmethod
    def resolve_token(*, token: str) -> ActionKind:
        """
        Resolve a raw action-type token to its functional kind.
        """

        try:
            return ActionKindResolver.resolve(action_type=ActionType(token.lower()))
        except ValueError:
            return ActionKind.UNKNOWN


class PastActionEntry(BaseModel):
    """
    Typed annotation for one prior action surfaced into the vision prompt.

    Built from the raw history dict already maintained by the memory/knowledge
    layer rather than reconstructing the full :class:`Action` schema, because
    the vision prompt sits downstream of that layer and only has the persisted
    fields. ``kind`` is derived from the recorded action-type token, which is
    authoritative for kind classification at serialization time.
    """

    model_config = ConfigDict(frozen=True)

    action: str = Field(description="Raw action_type token as recorded in history.")
    target: str = Field(
        default="",
        description="Action target string as recorded; empty when target was not captured.",
    )
    kind: ActionKind = Field(description="Functional category derived from ``action``.")
    expected_screen_change: bool = Field(
        description=(
            "True for NAVIGATION and INPUT, False for every other kind. Lets the "
            "model interpret a no-change outcome correctly when the action was "
            "not supposed to change the screen in the first place."
        ),
    )
    sub_goal_index: Optional[int] = Field(
        default=None,
        description="Sub-goal index this action was recorded under, when available.",
    )

    @classmethod
    def from_raw(cls, *, entry: Dict[str, Any]) -> "PastActionEntry":
        """
        Construct a typed entry from a raw history dict.

        Tolerant of missing fields: ``action`` falls back to "unknown", ``target``
        to empty string, ``sub_goal_index`` stays None. ``expected_screen_change``
        is derived from ``kind`` so it stays consistent regardless of which
        upstream layer produced the dict.
        """

        action_token = str(entry.get("action") or entry.get("action_type") or "unknown")

        kind = ActionKindResolver.resolve_token(token=action_token)
        expects_change = kind in (ActionKind.NAVIGATION, ActionKind.INPUT)

        sub_goal_index_raw = entry.get("sub_goal_index")
        sub_goal_index = (
            int(sub_goal_index_raw) if isinstance(sub_goal_index_raw, (int, float)) else None
        )

        return cls(
            kind=kind,
            action=action_token,
            sub_goal_index=sub_goal_index,
            expected_screen_change=expects_change,
            target=str(entry.get("target") or ""),
        )
