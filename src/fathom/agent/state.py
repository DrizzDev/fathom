from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Deque, Dict, List, Optional, Set

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


@dataclass
class LoopDetector:
    """
    Detects when agent is stuck in a loop.

    Uses a sliding window of screen hashes to detect repeated states.
    Implements exponential backoff for recovery attempts.
    """

    threshold: int = 3
    window_size: int = 5

    __max_recovery: int = 3
    __recovery_attempts: int = 0
    __recent_hashes: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    __recent_actions: Deque[str] = field(default_factory=lambda: deque(maxlen=5))

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

        Returns True if the same screen hash appears threshold times
        in the recent window, or if the same action is repeated.
        """

        if len(self.__recent_hashes) < self.threshold:
            return False

        # Check for repeated screens
        hash_counts: Dict[str, int] = {}
        for screen_hash in self.__recent_hashes:
            hash_counts[screen_hash] = hash_counts.get(screen_hash, 0) + 1
            if hash_counts[screen_hash] >= self.threshold:
                logger.warning(
                    f"Loop detected: Screen hash {screen_hash} seen {hash_counts[screen_hash]} times."
                )
                return True

        # Check for repeated actions
        if len(self.__recent_actions) >= self.threshold:
            action_counts: Dict[str, int] = {}
            for action_description in self.__recent_actions:
                action_counts[action_description] = action_counts.get(action_description, 0) + 1
                if action_counts[action_description] >= self.threshold:
                    logger.warning(
                        f"Loop detected: Action '{action_description}' repeated {action_counts[action_description]} times."
                    )
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


@dataclass
class ActionHistory:
    """
    Tracks action history for context building with token optimization.
    """

    max_size: int = 10
    __failure_count: int = 0
    __actions: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10))

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
            self.__failure_count += 1

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
        Legacy support for checkpointing and context building.
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


class AgentState:
    """
    Stateful agent context for planning and execution.

    Manages:
    - Screen state history with deduplication
    - Action history with success/failure tracking
    - Loop detection with recovery strategies
    - Context building for vision-language models

    Thread-safe for async operations. Serializable for checkpointing.
    """

    def __init__(
        self,
        intent: str,
        *,
        max_steps: int = 20,
        loop_threshold: int = 3,
        context_window: int = 10,
    ) -> None:
        """
        Initialize agent state.

        Args:
            intent: The goal to achieve.
            max_steps: Maximum steps before giving up.
            loop_threshold: Screen repetitions before stuck detection.
            context_window: Number of recent items to keep in context.
        """

        self.__step_count = 0
        self.__intent = intent
        self.__max_steps = max_steps

        self.__loop_detector = LoopDetector(
            threshold=loop_threshold,
            window_size=context_window,
        )
        self.__action_history = ActionHistory(max_size=context_window)

        self.__screen_hashes: Set[str] = set()
        self.__current_screen: Optional[ScreenState] = None

        self.__is_complete = False
        self.__completion_reason: Optional[str] = None

    @property
    def intent(self) -> str:
        """
        The goal being pursued.
        """

        return self.__intent

    @property
    def step_count(self) -> int:
        """
        Number of steps taken.
        """

        return self.__step_count

    @property
    def is_complete(self) -> bool:
        """
        Whether the intent has been achieved.
        """

        return self.__is_complete

    @property
    def is_stuck(self) -> bool:
        """
        Whether the agent is stuck in a loop.
        """

        return self.__loop_detector.is_stuck()

    @property
    def can_continue(self) -> bool:
        """
        Whether the agent can continue execution.
        """

        if self.__is_complete:
            return False

        if self.__step_count >= self.__max_steps:
            return False

        return not (self.is_stuck and not self.__loop_detector.can_recover())

    @property
    def current_screen(self) -> Optional[ScreenState]:
        """
        Current screen state.
        """

        return self.__current_screen

    def update_screen(self, screen: ScreenState) -> bool:
        """
        Update current screen state.

        Args:
            screen: New screen state.

        Returns:
            True if this is a new screen, False if seen before.
        """

        self.__current_screen = screen
        is_new_screen = screen.visual_hash not in self.__screen_hashes

        if is_new_screen:
            self.__screen_hashes.add(screen.visual_hash)
            logger.debug(f"New screen detected: {screen.visual_hash[:8]} ({screen.activity})")
        else:
            logger.debug(f"Returning to known screen: {screen.visual_hash[:8]}")

        self.__loop_detector.record(screen_hash=screen.visual_hash)
        return is_new_screen

    def record_step(self, result: StepResult) -> None:
        """Record a completed step.

        Args:
            result: Result of the executed step.
        """
        self.__step_count += 1

        # Optimized record including activity context
        activity = self.__current_screen.activity if self.__current_screen else "unknown"
        self.__action_history.record_action(
            action=result.step.action, success=result.success, activity=activity
        )

        if result.step.action.action_type == ActionType.COMPLETE and result.success:
            self.mark_complete(reason="Goal achieved via COMPLETE action")

    def mark_complete(self, reason: str) -> None:
        """
        Mark the intent as complete.

        Args:
            reason: Why the intent is considered complete.
        """

        self.__is_complete = True
        self.__completion_reason = reason

    def get_recovery_action(self) -> Optional[Action]:
        """
        Get a recovery action when stuck.

        Uses escalating recovery strategies:
        1. First attempt: Press back
        2. Second attempt: Scroll down
        3. Third attempt: Press home

        Returns:
            Recovery action or None if recovery exhausted.
        """

        if not self.__loop_detector.can_recover():
            return None

        attempt_number = self.__loop_detector.record_recovery_attempt()

        if attempt_number == 1:
            return Action(
                confidence=0.9,
                action_type=ActionType.BACK,
                target="system: back",
                rationale="Loop detected (Screen repeating). Forcing BACK to break context.",
            )
        elif attempt_number == 2:
            return Action(
                confidence=0.8,
                action_type=ActionType.SCROLL,
                target="system: scroll",
                rationale="Loop detected (Screen repeating). Forcing SCROLL to reveal new state.",
            )
        else:
            return Action(
                confidence=0.7,
                action_type=ActionType.HOME,
                target="system: home",
                rationale="Loop detected (Screen repeating). Forcing HOME to reset agent.",
            )

    def build_context(self) -> Dict[str, object]:
        """
        Build context for vision-language model with token optimization.
        """

        current_activity = self.__current_screen.activity if self.__current_screen else "unknown"

        return {
            "intent": self.__intent,
            "is_stuck": self.is_stuck,
            "max_steps": self.__max_steps,
            "step_count": self.__step_count,
            "unique_screens_seen": len(self.__screen_hashes),
            "compact_history": self.__action_history.get_compact_history(),
            "relevant_failures": self.__action_history.get_activity_failures(
                current_activity=current_activity
            ),
        }

    def should_avoid_action(self, action: Action) -> bool:
        """
        Check if an action should be avoided due to recent failures.

        Args:
            action: Proposed action.

        Returns:
            True if action has failed recently.
        """

        return self.__action_history.has_repeated_failure(action=action)

    def to_checkpoint(self) -> Dict[str, object]:
        """
        Serialize state for checkpointing.

        Returns:
            Dictionary suitable for JSON serialization.
        """

        return {
            "intent": self.__intent,
            "max_steps": self.__max_steps,
            "step_count": self.__step_count,
            "is_complete": self.__is_complete,
            "screen_hashes": list(self.__screen_hashes),
            "completion_reason": self.__completion_reason,
            "action_stats": self.__action_history.get_stats(),
            "action_context": self.__action_history.get_context(),
        }

    def __restore_from_data(
        self,
        step_count: int,
        is_complete: bool,
        screen_hashes: List[str],
        completion_reason: Optional[str],
    ) -> None:
        """
        Restore internal state from checkpoint data.
        """

        self.__step_count = step_count
        self.__is_complete = is_complete
        self.__completion_reason = completion_reason

        for screen_hash in screen_hashes:
            self.__screen_hashes.add(screen_hash)

    @classmethod
    def from_checkpoint(cls, data: Dict[str, object]) -> "AgentState":
        """
        Restore state from checkpoint.

        Args:
            data: Checkpoint data from to_checkpoint().

        Returns:
            Restored AgentState.
        """

        max_steps_value = data.get("max_steps")
        max_steps = int(max_steps_value) if isinstance(max_steps_value, (int, float)) else 20

        state = cls(intent=str(data["intent"]), max_steps=max_steps)

        step_count_value = data.get("step_count")
        step_count = int(step_count_value) if isinstance(step_count_value, (int, float)) else 0

        is_complete = bool(data.get("is_complete", False))

        reason_value = data.get("completion_reason")
        completion_reason = str(reason_value) if reason_value else None

        screen_hashes: List[str] = []
        hashes_value = data.get("screen_hashes")

        if isinstance(hashes_value, list):
            screen_hashes = [str(screen_hash) for screen_hash in hashes_value]

        state.__restore_from_data(
            step_count=step_count,
            is_complete=is_complete,
            screen_hashes=screen_hashes,
            completion_reason=completion_reason,
        )

        return state
