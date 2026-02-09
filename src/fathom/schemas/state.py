from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from fathom.schemas.actions import Action


class LoopDetector(BaseModel):
    """
    Detects when agent is stuck in a loop.

    Uses a sliding window of screen hashes to detect repeated states.
    Implements exponential backoff for recovery attempts.
    """

    threshold: int = Field(default=3, description="Screen repetition threshold")
    window_size: int = Field(default=5, description="Size of the sliding window")

    __max_recovery: int = PrivateAttr(default=3)
    __recovery_attempts: int = PrivateAttr(default=0)
    __recent_hashes: Deque[str] = PrivateAttr(default_factory=lambda: deque(maxlen=5))
    __recent_actions: Deque[str] = PrivateAttr(default_factory=lambda: deque(maxlen=5))

    def record(self, screen_hash: str, action_description: Optional[str] = None) -> None:
        """
        Record a screen hash and optionally an action description.
        """

        self.__recent_hashes.append(screen_hash)

        if action_description:
            self.__recent_actions.append(action_description)

    def is_stuck(self) -> bool:
        """
        Check if agent appears stuck in a loop.
        """

        if len(self.__recent_hashes) < self.threshold:
            return False

        # Check for repeated screens
        hash_counts: Dict[str, int] = {}
        for screen_hash in self.__recent_hashes:
            hash_counts[screen_hash] = hash_counts.get(screen_hash, 0) + 1
            if hash_counts[screen_hash] >= self.threshold:
                return True

        # Check for repeated actions
        if len(self.__recent_actions) >= self.threshold:
            action_counts: Dict[str, int] = {}
            for action_description in self.__recent_actions:
                action_counts[action_description] = action_counts.get(action_description, 0) + 1
                if action_counts[action_description] >= self.threshold:
                    return True

        return False

    def can_recover(self) -> bool:
        """
        Check if recovery is still possible.
        """

        return self.__recovery_attempts < self.__max_recovery

    def record_recovery_attempt(self) -> int:
        """
        Record a recovery attempt and return the attempt number.
        """

        self.__recovery_attempts += 1
        return self.__recovery_attempts

    def reset(self) -> None:
        """
        Reset loop detection state.
        """

        self.__recent_hashes.clear()
        self.__recent_actions.clear()

        self.__recovery_attempts = 0


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
