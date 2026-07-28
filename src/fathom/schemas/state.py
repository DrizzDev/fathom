from __future__ import annotations

from collections import deque
from logging import getLogger
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState

logger = getLogger(__name__)


class VerificationLoopState(BaseModel):
    """
    Serializable verifier-loop state for one same-screen verifier rejection streak.
    """

    recorded_step_count: int = Field(
        ge=0,
        description="Recorded action-step count at which the rejection streak began.",
    )
    activity: str = Field(description="Foreground activity observed during verification.")
    screen: Optional[ScreenState] = Field(
        default=None,
        description="Best-known screen state for the rejection streak, when available.",
    )
    consecutive_rejections: int = Field(
        ge=1,
        default=1,
        description="Number of consecutive verifier rejections in this streak.",
    )

    model_config = ConfigDict(frozen=True)

    def matches(
        self,
        *,
        activity: str,
        recorded_step_count: int,
        screen: Optional[ScreenState],
    ) -> bool:
        """
        Return whether a new verifier rejection belongs to this streak.
        """

        _ = recorded_step_count

        if activity != self.activity:
            return False

        if self.screen is not None and screen is not None:
            return self.screen.is_same_screen(other=screen)

        return False

    def next_rejection(
        self,
        *,
        activity: str,
        recorded_step_count: int,
        screen: Optional[ScreenState],
    ) -> "VerificationLoopState":
        """
        Return the next verifier-loop state after observing one rejection.
        """

        if not self.matches(
            screen=screen,
            activity=activity,
            recorded_step_count=recorded_step_count,
        ):
            return VerificationLoopState(
                screen=screen,
                activity=activity,
                recorded_step_count=recorded_step_count,
            )

        return self.model_copy(
            update={"consecutive_rejections": self.consecutive_rejections + 1},
        )


class LoopDetectorState(BaseModel):
    """
    Serializable snapshot of :class:`LoopDetector` internal deque's used
    to round-trip loop-detection evidence through graph checkpoints so accumulating signals survive iteration boundaries.
    """

    types: List[str] = Field(default_factory=list, description="Action type tokens window")
    actions: List[str] = Field(default_factory=list, description="Action descriptors window")
    hashes: List[str] = Field(default_factory=list, description="Screen visual hashes window")
    screens: List[ScreenState] = Field(
        default_factory=list, description="ScreenState entries window"
    )

    timestamps: List[float] = Field(default_factory=list, description="Record timestamps window")
    effect_statuses: List[str] = Field(
        default_factory=list,
        description=(
            "Per-record action-effect status tokens window. Empty string means the status was not recorded for that slot "
            "(legacy checkpoint or screen-only update). Kept as plain strings to keep the snapshot vendor-neutral."
        ),
    )
    recovery_attempts: int = Field(default=0, description="Autonomous recovery attempts taken")

    model_config = ConfigDict(frozen=True)


class InteractionTracker(BaseModel):
    """
    Elegantly tracks the cadence and repetition of agent interactions.

    Provides deterministic data on consecutive action sequences to enforce
    behavioral constraints (e.g., 'Don't swipe more than 3 times').
    """

    __consecutive_count: int = PrivateAttr(default=0)
    __last_action_type: Optional[str] = PrivateAttr(default=None)
    __total_counters: Dict[str, int] = PrivateAttr(default_factory=dict)

    def record(self, action_type: str) -> None:
        """
        Records an interaction and updates cadence metrics.
        """
        # Update Total
        self.__total_counters[action_type] = self.__total_counters.get(action_type, 0) + 1

        # Update Consecutive
        if action_type == self.__last_action_type:
            self.__consecutive_count += 1
        else:
            self.__last_action_type = action_type
            self.__consecutive_count = 1

    def get_cadence_note(self) -> Optional[str]:
        """
        Returns a semantic note about current interaction repetition if significant.
        Example: "Consecutive swipe_up: 3"
        """

        if self.__consecutive_count > 1 and self.__last_action_type:
            return f"Consecutive {self.__last_action_type}: {self.__consecutive_count}"

        return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns raw tracking data.
        """

        return {
            "last_type": self.__last_action_type,
            "totals": dict(self.__total_counters),
            "consecutive": self.__consecutive_count,
        }


class ActionHistory(BaseModel):
    """
    Tracks action history for context building with token optimization.
    """

    max_size: int = Field(default=10, description="Maximum history size")

    __failure_count: int = PrivateAttr(default=0)
    __actions: Deque[Dict[str, Any]] = PrivateAttr(default_factory=lambda: deque(maxlen=10))

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_ActionHistory__actions", deque(maxlen=self.max_size))

    def record_action(self, action: Action, success: bool, activity: str) -> None:
        """
        Record an action with its outcome and associated activity.
        """

        self.__actions.append(
            {
                "success": success,
                "activity": activity,
                "type": action.action_type.value.upper(),
                "full_description": action.to_description(),
                "target": action.natural_language_target or action.label_id or "UI",
            }
        )
        if not success:
            object.__setattr__(self, "_ActionHistory__failure_count", self.__failure_count + 1)

    def get_compact_history(self) -> str:
        """
        Returns a token-optimized representation of history.
        Format: TYPE:Target:Result
        """

        parts = []

        for action in self.__actions:
            result_indicator = "✓" if action["success"] else "✗"
            parts.append(f"{action['type']}:{action['target']}:{result_indicator}")

        return " | ".join(parts) if parts else "None"

    def get_context(self) -> List[str]:
        """
        Returns list of action descriptions.
        """

        return [action["full_description"] for action in self.__actions]

    def get_history_items(self) -> List[Dict[str, Any]]:
        """
        Returns raw list of history items for smart context.
        """

        return list(self.__actions)

    def recent_action_descriptors(self, *, count: int) -> List[str]:
        """
        Return the trailing ``count`` action descriptors in execution order.
        """

        if count <= 0:
            return []
        return [action["full_description"] for action in list(self.__actions)[-count:]]

    def get_activity_failures(self, current_activity: str) -> List[str]:
        """
        Returns only failures that occurred on the current activity.
        """

        return [
            f"{action['type']} on {action['target']}"
            for action in self.__actions
            if not action["success"] and action["activity"] == current_activity
        ]

    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about action history.
        """

        total_count = len(self.__actions)

        return {
            "total": total_count,
            "failure": self.__failure_count,
            "success": total_count - self.__failure_count,
        }

    def has_repeated_failure(self, action: Action) -> bool:
        """
        Check if this exact action has failed recently.
        """

        description = action.to_description()

        return any(
            not historical["success"] and historical["full_description"] == description
            for historical in self.__actions
        )
