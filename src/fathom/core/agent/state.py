from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import ActionHistory, InteractionTracker, LoopDetector
from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


class AgentState:
    """
    Stateful agent context for planning and execution.

    Manages:
    - Screen state history with deduplication
    - Action history with success/failure tracking
    - Interaction tracking for behavioral constraints
    - Loop detection with recovery strategies

    Thread-safe for async operations. Serializable for checkpointing.
    """

    def __init__(
        self,
        intent: str,
        *,
        max_steps: int = 20,
        loop_threshold: int = 3,
        context_window: int = 10,
        realignment_budget: int = 3,
    ) -> None:
        """
        Initialize agent state.

        Args:
            intent: The goal to achieve.
            max_steps: Maximum steps before giving up.
            loop_threshold: Screen repetitions before stuck detection.
            context_window: Number of recent items to keep in context.
            realignment_budget: Maximum human interventions allowed for loops.
        """

        self.__step_count = 0
        self.__intent = intent
        self.__max_steps = max_steps
        self.__realignment_budget = realignment_budget

        self.__loop_detector = LoopDetector(
            threshold=loop_threshold,
            window_size=context_window,
        )
        self.__action_history = ActionHistory(max_size=context_window)
        self.__interaction_tracker = InteractionTracker()

        self.__seen_screens: List[ScreenState] = []
        self.__current_screen: Optional[ScreenState] = None

        self.__is_complete = False
        self.__completion_reason: Optional[str] = None

        # Enhanced state fields
        self.__knowledge: Dict[str, Any] = {}
        self.__current_screen_name: Optional[str] = None

        self.__last_error: Optional[str] = None
        self.__last_action_type: Optional[str] = None
        self.__last_action_description: Optional[str] = None

        # HITL Tracking
        self.__realignment_count = 0

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
    def completion_reason(self) -> Optional[str]:
        """
        Returns Completion Reason
        """

        return self.__completion_reason

    @property
    def last_action_type(self) -> Optional[str]:
        """
        The type of the most recently executed action, if any.
        """

        return self.__last_action_type

    @property
    def is_stuck(self) -> bool:
        """
        Whether the agent is stuck in a loop.
        """

        return self.__loop_detector.is_stuck()

    def record_hitl_intervention(self) -> None:
        """
        Atomic update when user provides guidance to break a loop.
        """

        self.__realignment_count += 1
        self.__loop_detector.reset()

    @property
    def can_continue(self) -> bool:
        """
        Whether the agent can continue execution.
        """

        if self.__is_complete:
            return False

        if self.__step_count >= self.__max_steps:
            return False

        # If stuck, evaluate based on recovery mode
        if self.is_stuck:
            # We fail if BOTH budgets are exhausted or relevant budget is exhausted.
            # In interactive mode, we only care about realignment budget.
            if self.__realignment_count >= self.__realignment_budget:
                return False

            # Autonomous budget (used in non-interactive mode)
            return bool(self.__loop_detector.can_recover())

        return True

    @property
    def current_screen(self) -> Optional[ScreenState]:
        """
        Current screen state.
        """

        return self.__current_screen

    @property
    def tracking_note(self) -> Optional[str]:
        """
        Provides semantic feedback on interaction cadence.
        """

        return self.__interaction_tracker.get_cadence_note()

    def __is_new_screen(self, screen: ScreenState) -> bool:
        """
        Check if screen is new.
        """

        return all(not seen.is_same_screen(screen) for seen in self.__seen_screens)

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
        is_new_screen = self.__is_new_screen(screen=screen)

        if is_new_screen:
            self.__seen_screens.append(screen)
            logger.debug(f"New screen detected: {screen.visual_hash[:8]} ({screen.activity})")
            # If we reached a truly new screen (never seen in session), we have made progress.
            self.__loop_detector.reset()
        elif previous_screen and previous_screen.activity_hash != screen.activity_hash:
            # Keep loop history when revisiting known screens across activities.
            # This preserves oscillation/stall evidence instead of masking it.
            logger.debug(
                (
                    f"Activity changed on known screen: {previous_screen.activity} -> "
                    f"{screen.activity}. Preserving loop detector state."
                )
            )
        else:
            logger.debug(f"Returning to known screen: {screen.visual_hash[:8]}")

        self.__loop_detector.record(
            screen=screen,
            action_type=self.__last_action_type,
            action_description=self.__last_action_description,
        )
        return is_new_screen

    def set_knowledge(self, key: str, value: Any) -> None:
        """
        Set a fact in knowledge base.
        """

        self.__knowledge[key] = value

    def set_last_error(self, error: str) -> None:
        """
        Set the last error message.
        """

        self.__last_error = error

    def get_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Returns recent action history as structured items.
        """

        return self.__action_history.get_history_items()[-limit:]

    def record_step(self, result: StepResult) -> None:
        """
        Record a completed step.

        Args:
            result: Result of the executed step.
        """

        self.__step_count += 1

        # Optimized record including activity context
        activity = self.__current_screen.activity if self.__current_screen else "unknown"
        self.__action_history.record_action(
            action=result.step.action, success=result.success, activity=activity
        )
        self.__interaction_tracker.record(action_type=result.step.action.action_type.value)

        self.__last_action_type = result.step.action.action_type.value
        self.__last_action_description = result.step.action.to_description()

        if result.step.action.action_type == ActionType.COMPLETE and result.success:
            self.mark_complete(reason=CompletionReason.SUCCESS.value)

    def mark_complete(self, reason: str) -> None:
        """
        Mark the intent as complete.

        Args:
            reason: Why the intent is considered complete.
        """

        self.__is_complete = True
        self.__completion_reason = reason

    def reset_completion(self) -> None:
        """
        Clear the completion state.
        """

        self.__is_complete = False
        self.__completion_reason = None

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
        Record a planner-level recovery attempt.
        """

        return self.__loop_detector.record_recovery_attempt()

    def reset_loop_detector(self) -> None:
        """
        Reset loop detector state after an explicit progress signal.
        """

        self.__loop_detector.signal_content_exhausted()

    def get_delta_context(self) -> Dict[str, object]:
        """
        Return compact no-XML delta context used for planning hints.
        """

        return {"last_delta_score": None, "low_delta_streak": 0}

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
            "delta_context": self.get_delta_context(),
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
            "realignment_count": self.__realignment_count,
            "completion_reason": self.__completion_reason,
            "realignment_budget": self.__realignment_budget,
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
        *,
        realignment_count: int = 0,
    ) -> None:
        """
        Restore internal state from checkpoint data.
        """

        self.__step_count = step_count
        self.__is_complete = is_complete
        self.__completion_reason = completion_reason
        self.__realignment_count = realignment_count

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
        max_steps = (
            int(cast("int", max_steps_value)) if isinstance(max_steps_value, (int, float)) else 20
        )

        realignment_budget_value = data.get("realignment_budget")
        realignment_budget = (
            int(cast("int", realignment_budget_value))
            if isinstance(realignment_budget_value, (int, float))
            else 3
        )

        state = cls(
            intent=str(data["intent"]), max_steps=max_steps, realignment_budget=realignment_budget
        )

        step_count_value = data.get("step_count")
        step_count = (
            int(cast("int", step_count_value)) if isinstance(step_count_value, (int, float)) else 0
        )

        is_complete = bool(data.get("is_complete", False))
        realignment_count = int(cast("int", data.get("realignment_count", 0)))

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
            realignment_count=realignment_count,
        )

        return state
