from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, Set

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import ActionHistory, LoopDetector
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


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
        max_steps: int = 100,
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

        self.__seen_screens: List[ScreenState] = []
        self.__current_screen: Optional[ScreenState] = None

        self.__is_complete = False
        self.__completion_reason: Optional[str] = None

        # Enhanced state fields
        self.__knowledge: Dict[str, Any] = {}
        self.__current_screen_name: Optional[str] = None
        self.__last_error: Optional[str] = None
        self.__last_action_description: Optional[str] = None
        self.__last_action_type: Optional[ActionType] = None

        # Non-physical actions that don't change the screen
        self.__non_physical_actions: Set[ActionType] = {
            ActionType.WAIT,
            ActionType.COMPLETE,
            ActionType.SAVE_MEMORY,
            ActionType.RETRIEVE_MEMORY,
        }

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

    def __is_new_screen(self, screen: ScreenState) -> bool:
        """
        Check if screen is new.
        """

        return all(not seen.is_same_screen(screen) for seen in self.__seen_screens)

    def reset_loop_detector(self) -> None:
        """
        Reset loop detector (e.g., after content exhaustion signal).
        """

        self.__loop_detector.signal_content_exhausted()

    @property
    def last_action_type(self) -> Optional[ActionType]:
        """
        The type of the last action executed.
        """
        return self.__last_action_type

    def update_screen(self, screen: ScreenState) -> bool:
        """
        Update current screen state.

        Args:
            screen: New screen state.

        Returns:
            True if this is a new screen, False if seen before.
        """

        previous_screen = self.__current_screen
        self.__current_screen = screen
        # Fuzzy matching for seen screens
        is_new_screen = self.__is_new_screen(screen)

        logger.debug(
            f"[H2] Screen update classification | "
            f"is_new={is_new_screen} seen_count={len(self.__seen_screens)} "
            f"current={screen.activity} hash={screen.activity_hash} "
            f"prev={previous_screen.activity if previous_screen else None}"
        )

        if is_new_screen:
            self.__seen_screens.append(screen)
            logger.debug(f"New screen detected: {screen.visual_hash[:8]} ({screen.activity})")
            # If we reached a new screen, we are definitely not stuck in a local loop.
            self.__loop_detector.reset()
        elif previous_screen and previous_screen.activity_hash != screen.activity_hash:
            # Activity changed but screen was seen before (e.g., revisiting a page via
            # a different navigation path). This is progress, not a loop.
            logger.debug(
                f"Activity changed: {previous_screen.activity} -> {screen.activity}. "
                f"Resetting loop detector."
            )
            self.__loop_detector.reset()
        else:
            logger.debug(f"Returning to known screen: {screen.visual_hash[:8]}")

        # Only feed the loop detector if the last action was physical.
        # Non-physical actions (WAIT, COMPLETE, memory ops) don't change
        # the screen, so recording them inflates the repeat count and
        # causes false-positive stuck detection (e.g., validate intents).
        if self.__last_action_type not in self.__non_physical_actions:
            self.__loop_detector.record(
                screen=screen, action_description=self.__last_action_description
            )
        else:
            logger.debug(
                f"Skipping loop detector record for non-physical action: {self.__last_action_type}"
            )
        return is_new_screen

    def set_knowledge(self, key: str, value: Any) -> None:
        """Set a fact in knowledge base."""
        self.__knowledge[key] = value

    def set_last_error(self, error: str) -> None:
        """Set the last error message."""
        self.__last_error = error

    def get_smart_context(self, max_history: int = 5) -> str:
        """
        Structured context for LLM — current state + recent history + errors.
        Ported from interactive_testing state manager.
        """
        lines = ["=== CURRENT STATE ==="]

        if self.__current_screen:
            # Use activity name if available, or just generic
            screen_name = self.__current_screen.activity or "Unknown Screen"
            lines.append(f"Current Screen: {screen_name}")

        # Known knowledge
        if self.__knowledge:
            lines.append("Known Facts:")
            for key, value in self.__knowledge.items():
                lines.append(f"- {key}: {value}")

        lines.append(f"\n=== RECENT HISTORY (Last {max_history}) ===")
        # Get recent items from action history
        recent = self.__action_history.get_history_items()[-max_history:]

        for index, item in enumerate(recent):
            status = "[OK]" if item["success"] else "[FAIL]"
            action_description = f"{item['type']}:{item['target']}"
            lines.append(f"{index + 1}. {status} {action_description}")

        if self.__last_error:
            lines.append(f"\n[WARN] LAST ERROR: {self.__last_error}")

        lines.append("=== END STATE ===")
        return "\n".join(lines)

    def record_step(self, result: StepResult) -> None:
        """
        Record a completed step.

        Args:
            result: Result of the executed step.
        """

        self.__step_count += 1
        self.__last_action_type = result.step.action.action_type

        # Optimized record including activity context
        activity = self.__current_screen.activity if self.__current_screen else "unknown"
        self.__action_history.record_action(
            action=result.step.action,
            success=result.success,
            activity=activity,
            screen_changed=result.screen_changed,
        )
        self.__last_action_description = result.step.action.to_description()

        logger.debug(
            f"[H7] Recorded executed action | "
            f"step={self.__step_count} action={result.step.action.to_description()} "
            f"type={result.step.action.action_type.value} success={result.success}"
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
                target="system: back",
                action_type=ActionType.BACK,
                rationale="Loop detected (Screen repeating). Forcing BACK to break context.",
            )
        elif attempt_number == 2:
            return Action(
                confidence=0.8,
                target="system: scroll",
                action_type=ActionType.SCROLL,
                rationale="Loop detected (Screen repeating). Forcing SCROLL to reveal new state.",
            )
        else:
            return Action(
                confidence=0.7,
                target="system: home",
                action_type=ActionType.HOME,
                rationale="Loop detected (Screen repeating). Forcing HOME to reset agent.",
            )

    def record_recovery_attempt(self) -> int:
        """
        Record a recovery attempt (prompt-driven).
        """
        return self.__loop_detector.record_recovery_attempt()

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
            "unique_screens_seen": len(self.__seen_screens),
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
            "completion_reason": self.__completion_reason,
            "action_stats": self.__action_history.get_stats(),
            "action_context": self.__action_history.get_context(),
            "seen_screens": [screen.model_dump() for screen in self.__seen_screens],
        }

    def __restore_from_data(
        self,
        step_count: int,
        is_complete: bool,
        completion_reason: Optional[str],
        seen_screens: List[Dict[str, Any]],
    ) -> None:
        """
        Restore internal state from checkpoint data.
        """

        self.__step_count = step_count
        self.__is_complete = is_complete
        self.__completion_reason = completion_reason

        for data in seen_screens:
            self.__seen_screens.append(ScreenState(**data))

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

        seen_screens: List[Dict[str, Any]] = []
        screens_value = data.get("seen_screens")

        if isinstance(screens_value, list):
            seen_screens = [dict(screen) for screen in screens_value]

        state.__restore_from_data(
            step_count=step_count,
            is_complete=is_complete,
            seen_screens=seen_screens,
            completion_reason=completion_reason,
        )

        return state
