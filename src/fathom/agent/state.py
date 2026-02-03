from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from logging import getLogger
from typing import Deque, Dict, List, Optional, Set

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


@dataclass
class LoopDetector:
    """Detects when agent is stuck in a loop.

    Uses a sliding window of screen hashes to detect repeated states.
    Implements exponential backoff for recovery attempts.
    """

    window_size: int = 5
    threshold: int = 3
    __recent_hashes: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    __recent_actions: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    __recovery_attempts: int = 0
    __max_recovery: int = 3

    def record(self, screen_hash: str, action_desc: Optional[str] = None) -> None:
        """
        Record a screen hash and optionally an action description.
        """
        self.__recent_hashes.append(screen_hash)
        if action_desc:
            self.__recent_actions.append(action_desc)

    def is_stuck(self) -> bool:
        """Check if agent appears stuck in a loop.

        Returns True if the same screen hash appears threshold times
        in the recent window, or if the same action is repeated.
        """
        if len(self.__recent_hashes) < self.threshold:
            return False

        # Check for repeated screens
        hash_counts: Dict[str, int] = {}
        for h in self.__recent_hashes:
            hash_counts[h] = hash_counts.get(h, 0) + 1
            if hash_counts[h] >= self.threshold:
                logger.warning(f"Loop detected: Screen hash {h} seen {hash_counts[h]} times.")
                return True

        # Check for repeated actions
        if len(self.__recent_actions) >= self.threshold:
            action_counts: Dict[str, int] = {}
            for a in self.__recent_actions:
                action_counts[a] = action_counts.get(a, 0) + 1
                if action_counts[a] >= self.threshold:
                    logger.warning(
                        f"Loop detected: Action '{a}' repeated {action_counts[a]} times."
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
        self.__recovery_attempts = 0


@dataclass
class ActionHistory:
    """Tracks action history for context building.

    Maintains bounded history of actions and their outcomes
    to provide context for planning.
    """

    max_size: int = 10
    __actions: Deque[str] = field(default_factory=lambda: deque(maxlen=10))
    __failures: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    __success_count: int = 0
    __failure_count: int = 0

    def record_action(self, action: Action, success: bool) -> None:
        """
        Record an action and its outcome.
        """
        description = action.to_description()
        if success:
            self.__actions.append(f"✓ {description}")
            self.__success_count += 1
        else:
            self.__actions.append(f"✗ {description}")
            self.__failures.append(description)
            self.__failure_count += 1

    def get_context(self) -> List[str]:
        """
        Get recent actions as context for planning.
        """
        return list(self.__actions)

    def get_failures(self) -> List[str]:
        """
        Get recent failures for recovery planning.
        """
        return list(self.__failures)

    def get_stats(self) -> Dict[str, int]:
        """
        Get action statistics.
        """
        return {
            "success": self.__success_count,
            "failure": self.__failure_count,
            "total": self.__success_count + self.__failure_count,
        }

    def has_repeated_failure(self, action: Action) -> bool:
        """
        Check if this action has failed recently.
        """
        description = action.to_description()
        return description in self.__failures


class AgentState:
    """Stateful agent context for planning and execution.

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
        """Initialize agent state.

        Args:
            intent: The goal to achieve.
            max_steps: Maximum steps before giving up.
            loop_threshold: Screen repetitions before stuck detection.
            context_window: Number of recent items to keep in context.
        """
        self.__intent = intent
        self.__max_steps = max_steps
        self.__step_count = 0

        self.__loop_detector = LoopDetector(
            window_size=context_window,
            threshold=loop_threshold,
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
        """Update current screen state.

        Args:
            screen: New screen state.

        Returns:
            True if this is a new screen, False if seen before.
        """
        self.__current_screen = screen
        is_new = screen.visual_hash not in self.__screen_hashes

        if is_new:
            self.__screen_hashes.add(screen.visual_hash)
            logger.debug(f"New screen detected: {screen.visual_hash[:8]} ({screen.activity})")
        else:
            logger.debug(f"Returning to known screen: {screen.visual_hash[:8]}")

        self.__loop_detector.record(screen.visual_hash)
        return is_new

    def record_step(self, result: StepResult) -> None:
        """Record a completed step.

        Args:
            result: Result of the executed step.
        """
        self.__step_count += 1
        description = result.step.action.to_description()
        self.__action_history.record_action(result.step.action, result.success)

        # Feed action to loop detector for action-based loop detection
        self.__loop_detector.record(screen_hash=result.pre_hash, action_desc=description)

        if result.step.action.action_type == ActionType.COMPLETE and result.success:
            self.mark_complete("Goal achieved via COMPLETE action")

    def mark_complete(self, reason: str) -> None:
        """Mark the intent as complete.

        Args:
            reason: Why the intent is considered complete.
        """
        self.__is_complete = True
        self.__completion_reason = reason

    def get_recovery_action(self) -> Optional[Action]:
        """Get a recovery action when stuck.

        Uses escalating recovery strategies:
        1. First attempt: Press back
        2. Second attempt: Scroll down
        3. Third attempt: Press home

        Returns:
            Recovery action or None if recovery exhausted.
        """
        if not self.__loop_detector.can_recover():
            return None

        attempt = self.__loop_detector.record_recovery_attempt()

        if attempt == 1:
            return Action(
                confidence=0.5,
                action_type=ActionType.BACK,
                target="recovery: escape current screen",
                rationale="Stuck in loop, attempting navigation back",
            )
        elif attempt == 2:
            return Action(
                confidence=0.4,
                action_type=ActionType.SCROLL,
                target="recovery: reveal hidden content",
                rationale="Stuck in loop, attempting scroll to reveal new elements",
            )
        else:
            return Action(
                confidence=0.3,
                action_type=ActionType.HOME,
                target="recovery: reset to launcher",
                rationale="Stuck in loop, resetting to home screen",
            )

    def build_context(self) -> Dict[str, object]:
        """
        Build context for vision-language model.

        Returns:
            Dictionary with intent, history, failures, and state info.
        """

        return {
            "intent": self.__intent,
            "is_stuck": self.is_stuck,
            "max_steps": self.__max_steps,
            "step_count": self.__step_count,
            "unique_screens_seen": len(self.__screen_hashes),
            "action_stats": self.__action_history.get_stats(),
            "recent_actions": self.__action_history.get_context(),
            "recent_failures": self.__action_history.get_failures(),
        }

    def should_avoid_action(self, action: Action) -> bool:
        """
        Check if an action should be avoided due to recent failures.

        Args:
            action: Proposed action.

        Returns:
            True if action has failed recently.
        """

        return self.__action_history.has_repeated_failure(action)

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
        completion_reason: Optional[str],
        screen_hashes: List[str],
    ) -> None:
        """
        Restore internal state from checkpoint data.
        """

        self.__step_count = step_count
        self.__is_complete = is_complete
        self.__completion_reason = completion_reason

        for h in screen_hashes:
            self.__screen_hashes.add(h)

    @classmethod
    def from_checkpoint(cls, data: Dict[str, object]) -> "AgentState":
        """
        Restore state from checkpoint.

        Args:
            data: Checkpoint data from to_checkpoint().

        Returns:
            Restored AgentState.
        """

        max_steps_val = data.get("max_steps")
        max_steps = int(max_steps_val) if isinstance(max_steps_val, (int, float)) else 20

        state = cls(intent=str(data["intent"]), max_steps=max_steps)

        step_count_val = data.get("step_count")
        step_count = int(step_count_val) if isinstance(step_count_val, (int, float)) else 0

        is_complete = bool(data.get("is_complete", False))

        reason_val = data.get("completion_reason")
        completion_reason = str(reason_val) if reason_val else None

        screen_hashes: List[str] = []
        hashes_val = data.get("screen_hashes")

        if isinstance(hashes_val, list):
            screen_hashes = [str(h) for h in hashes_val]

        state.__restore_from_data(step_count, is_complete, completion_reason, screen_hashes)

        return state
