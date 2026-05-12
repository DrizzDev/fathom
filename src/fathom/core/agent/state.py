from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants import ActionType
from fathom.constants.reasoning import LOW_DELTA_PROGRESS_THRESHOLD
from fathom.constants.state import CompletionReason
from fathom.schemas.actions import Action
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.delta import DeltaSignal
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import ActionHistory, InteractionTracker, LoopDetector
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import SubGoal, SubGoalStatus

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
        self.__last_delta_score: Optional[float] = None
        self.__low_delta_streak: int = 0

        # Multi-turn rejection history for cross-iteration feedback loops.
        # Stores provider-neutral ConversationTurn objects so the next
        # vision.analyze() call can pass them as conversation_history.
        self.__rejection_history: Optional[List[ConversationTurn]] = None

        # HITL Tracking
        self.__realignment_count = 0

        # Sub-goal tracking for sequential intent execution
        self.__sub_goals: List[SubGoal] = []
        self.__current_sub_goal_index: int = 0
        self.__sub_goal_start_screen: Optional[str] = None  # Track screen hash when sub-goal starts
        self.__sub_goal_action_count: int = 0  # Track actions executed for current sub-goal

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

    @property
    def last_delta_score(self) -> Optional[float]:
        """
        Most recent post-action screen-change magnitude in [0.0, 1.0].
        """

        return self.__last_delta_score

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
            # New screen = visual progress. Clear screen-based detection but
            # preserve action history so action-repeat detection survives across
            # visually-different screens (e.g. tapping a counter button).
            self.__loop_detector.advance()
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

    def set_sub_goals(self, sub_goals: List[SubGoal]) -> None:
        """
        Set the decomposed sub-goals for this intent.

        Args:
            sub_goals: List of sequential sub-goals to execute.
        """
        self.__sub_goals = [goal.model_copy(deep=True) for goal in sub_goals]
        self.__current_sub_goal_index = 0
        if self.__sub_goals:
            self.__sub_goals[0].mark_in_progress()
            # Track starting screen for first sub-goal
            if self.__current_screen:
                self.__sub_goal_start_screen = self.__current_screen.visual_hash
            self.__sub_goal_action_count = 0
            logger.info(
                f"[AgentState] Initialized with {len(self.__sub_goals)} sub-goals. "
                f"Starting with: {self.__sub_goals[0].description}"
            )

    def get_current_sub_goal(self) -> Optional[SubGoal]:
        """
        Get the currently active sub-goal.

        Returns:
            Current sub-goal or None if no sub-goals defined.
        """
        if not self.__sub_goals or self.__current_sub_goal_index >= len(self.__sub_goals):
            return None
        return self.__sub_goals[self.__current_sub_goal_index]

    def set_current_sub_goal_index(self, index: int) -> None:
        """
        Set the current sub-goal index (used for checkpoint restore).

        Args:
            index: Index to set (will be clamped to valid range)
        """
        if self.__sub_goals:
            # Clamp to valid range
            clamped_index = max(0, min(index, len(self.__sub_goals) - 1))
            # Update index and mark the restored goal as in progress
            self.__current_sub_goal_index = clamped_index
            if clamped_index < len(self.__sub_goals):
                self.__sub_goals[clamped_index].mark_in_progress()
                logger.info(f"[AgentState] Sub-goal index restored to {clamped_index}")

    @property
    def sub_goal_list(self) -> List[SubGoal]:
        """
        List of all sub-goals.
        """
        return self.__sub_goals

    @property
    def current_sub_goal_index(self) -> int:
        """
        Current sub-goal index.
        """
        return self.__current_sub_goal_index

    def mark_current_sub_goal_complete(
        self,
        completion_signal: SubGoalCompletionSignal,
    ) -> bool:
        """
        Mark the current sub-goal as complete with multi-signal verification.

        Args:
            completion_signal: Multi-signal verification data

        Returns:
            True if advanced to next sub-goal, False if all complete.
        """
        current = self.get_current_sub_goal()
        if not current:
            return False

        # NOTE: Completion gating happens in planner with two-signal policy (llm + rationale).
        # This method just records the signals and advances. Trace verification is disabled
        # to prevent false positives from screen changes unrelated to sub-goal completion.
        updated_signal = SubGoalCompletionSignal(
            evidence=completion_signal.evidence,
            keyword_match=completion_signal.keyword_match,
            llm_confidence=completion_signal.llm_confidence,
            action_executed=completion_signal.action_executed,
            flagged_complete=completion_signal.flagged_complete,
            rationale_verified=completion_signal.rationale_verified,
            trace_verified=False,
        )

        # Mark complete with all signals
        current.mark_complete(
            trace_verified=updated_signal.trace_verified,
            flagged_complete=updated_signal.flagged_complete,
            rationale_verified=updated_signal.rationale_verified,
        )

        signal_count = updated_signal.count_signals()
        logger.info(
            f"[AgentState] Sub-goal {current.index} marked complete: {current.description} | "
            f"Signals: {signal_count} [llm={updated_signal.flagged_complete}, "
            f"trace={updated_signal.trace_verified}, rationale={updated_signal.rationale_verified}] | "
            f"Evidence: {updated_signal.evidence}"
        )

        # Advance to next sub-goal
        self.__current_sub_goal_index += 1

        if self.__current_sub_goal_index < len(self.__sub_goals):
            next_goal = self.__sub_goals[self.__current_sub_goal_index]
            next_goal.mark_in_progress()

            # Reset per-sub-goal counters on advancement.
            if self.__current_screen:
                self.__sub_goal_start_screen = self.__current_screen.visual_hash

            self.__sub_goal_action_count = 0
            logger.info(
                f"[AgentState] Advanced to sub-goal {next_goal.index}: {next_goal.description}"
            )
            return True
        else:
            logger.info("[AgentState] All sub-goals complete")
            return False

    def record_sub_goal_action(self) -> None:
        """
        Record that an action was executed for the current sub-goal.
        Call this from planner after executing each action.
        """
        self.__sub_goal_action_count += 1

    def replan_pending_sub_goals(self, *, new_sub_goals: List[SubGoal]) -> None:
        """
        Replace every unfinished sub-goal with ``new_sub_goals`` (preserving completed work)
        and mark the first new sub-goal IN_PROGRESS.
        """

        completed = [goal for goal in self.__sub_goals if goal.is_complete()]

        reindexed = [
            goal.model_copy(update={"index": len(completed) + offset})
            for offset, goal in enumerate(new_sub_goals)
        ]

        self.__sub_goal_action_count = 0
        self.__sub_goals = completed + reindexed
        self.__current_sub_goal_index = len(completed)

        if self.__current_sub_goal_index < len(self.__sub_goals):
            current = self.__sub_goals[self.__current_sub_goal_index]
            current.mark_in_progress()
            logger.info(
                f"[AgentState] Replanned: kept={len(completed)} replaced={len(reindexed)} "
                f"current={current.description[:60]!r}"
            )

    def all_sub_goals_complete(self) -> bool:
        """
        Check if all sub-goals have been completed.

        Returns:
            True if all sub-goals are complete or no sub-goals defined.
        """

        if not self.__sub_goals:
            return True

        return all(sg.is_complete() for sg in self.__sub_goals)

    def get_sub_goal_progress(self) -> tuple[int, int]:
        """
        Get current progress through sub-goals.

        Returns:
            Tuple of (current_index, total_count).
        """

        if not self.__sub_goals:
            return (0, 0)

        return (self.__current_sub_goal_index, len(self.__sub_goals))

    def get_all_sub_goals(self) -> List[SubGoal]:
        """
        Get all sub-goals.

        Returns:
            List of all sub-goals.
        """

        return self.__sub_goals.copy()

    def has_sub_goals(self) -> bool:
        """
        Check if sub-goals are defined.

        Returns:
            True if sub-goals exist.
        """

        return len(self.__sub_goals) > 0

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

        return {
            "last_delta_score": self.__last_delta_score,
            "low_delta_streak": self.__low_delta_streak,
        }

    def update_delta_context(self, delta: Optional[DeltaSignal]) -> None:
        """
        Update rolling delta metrics from the model's semantic delta signal.
        """

        if (score := self.__derive_delta_score(delta=delta)) is None:
            return

        self.__last_delta_score = score
        self.__low_delta_streak = (
            self.__low_delta_streak + 1 if score < LOW_DELTA_PROGRESS_THRESHOLD else 0
        )

    @staticmethod
    def __derive_delta_score(*, delta: Optional[DeltaSignal]) -> Optional[float]:
        """
        Project a :class:`DeltaSignal` onto a [0.0, 1.0] score; None when absent.
        """

        if delta is None:
            return None

        if delta.delta_confidence is not None:
            return max(0.0, min(1.0, float(delta.delta_confidence or 0)))

        if delta.delta_observed is True:
            return 1.0

        if delta.delta_observed is False:
            return 0.0

        return None

    def build_context(self) -> Dict[str, object]:
        """
        Build context for vision-language model with token optimization.
        """

        current_activity = self.__current_screen.activity if self.__current_screen else "unknown"

        context: Dict[str, object] = {
            "intent": self.__intent,
            "is_stuck": self.is_stuck,
            "max_steps": self.__max_steps,
            "step_count": self.__step_count,
            "delta_context": self.get_delta_context(),
            "unique_screens_seen": len(self.__seen_screens),
            "compact_history": self.__action_history.get_compact_history(),
            "relevant_failures": self.__action_history.get_activity_failures(
                current_activity=current_activity
            ),
        }

        return context

    def should_avoid_action(self, action: Action) -> bool:
        """
        Check if an action should be avoided due to recent failures.

        Args:
            action: Proposed action.

        Returns:
            True if action has failed recently.
        """

        return self.__action_history.has_repeated_failure(action=action)

    @property
    def rejection_history(self) -> Optional[List[ConversationTurn]]:
        """
        Returns stored multi-turn rejection history for cross-iteration feedback.
        """

        return self.__rejection_history

    def set_rejection_history(self, history: List[ConversationTurn]) -> None:
        """
        Store multi-turn rejection history so the next vision.analyze() cycle
        can pass it as conversation_history to the LLM.
        """

        self.__rejection_history = history

    def clear_rejection_history(self) -> None:
        """
        Clear rejection history after a successful action execution.
        """

        self.__rejection_history = None

    def record_repeated_action_failure(self, action: Action) -> None:
        """
        Mark a repeated action as a failure so it appears in the LLM's failure context.
        This prevents the model from proposing the same ineffective action on retry.
        """

        activity = self.__current_screen.activity if self.__current_screen else "unknown"
        self.__action_history.record_action(action=action, success=False, activity=activity)
        self.set_last_error(
            f"Action '{action.to_description()[:80]}' was repeated 3+ times on the same screen "
            "without progress. Try a different approach to achieve the same goal."
        )

    def is_action_repeating_on_screen(self, action: Action) -> bool:
        """
        Check if a tap/type action has been executed 3+ times on the current screen.
        Triggers replanning to break out of ineffective action loops.

        Args:
            action: Proposed action.

        Returns:
            True if the same action has been repeated 3+ times on the current screen.
        """

        if not self.__current_screen:
            return False

        return self.__loop_detector.has_repeated_action_on_same_screen(
            action_description=action.to_description(),
            screen_hash=self.__current_screen.visual_hash,
        )

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
            "last_delta_score": self.__last_delta_score,
            "low_delta_streak": self.__low_delta_streak,
            "action_stats": self.__action_history.get_stats(),
            "action_context": self.__action_history.get_context(),
            "seen_screens": [screen.model_dump() for screen in self.__seen_screens],
            "sub_goals": [goal.model_dump(mode="json") for goal in self.__sub_goals],
            "current_sub_goal_index": self.__current_sub_goal_index,
        }

    def __restore_from_data(
        self,
        step_count: int,
        is_complete: bool,
        completion_reason: Optional[str],
        seen_screens: List[Dict[str, Any]],
        *,
        low_delta_streak: int = 0,
        realignment_count: int = 0,
        current_sub_goal_index: int = 0,
        last_delta_score: Optional[float] = None,
        sub_goals: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Restore internal state from checkpoint data.
        """

        self.__step_count = step_count
        self.__is_complete = is_complete
        self.__completion_reason = completion_reason
        self.__realignment_count = realignment_count

        self.__last_delta_score = last_delta_score
        self.__low_delta_streak = max(0, low_delta_streak)

        for data in seen_screens:
            self.__seen_screens.append(ScreenState(**data))

        self.__sub_goals = []

        if sub_goals:
            for goal in sub_goals:
                self.__sub_goals.append(SubGoal.model_validate(goal))

        if self.__sub_goals:
            self.__current_sub_goal_index = min(
                max(0, current_sub_goal_index), len(self.__sub_goals)
            )
            if self.__current_sub_goal_index < len(self.__sub_goals):
                current = self.__sub_goals[self.__current_sub_goal_index]
                if current.status == SubGoalStatus.PENDING:
                    current.mark_in_progress()
        else:
            self.__current_sub_goal_index = 0

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
        low_delta_streak = int(cast("int", data.get("low_delta_streak", 0)))

        reason_value = data.get("completion_reason")
        completion_reason = str(reason_value) if reason_value else None
        last_delta_raw = data.get("last_delta_score")
        last_delta_score = (
            float(cast("float", last_delta_raw))
            if isinstance(last_delta_raw, (int, float))
            else None
        )

        seen_screens: List[Dict[str, Any]] = []
        screens_value = data.get("seen_screens")

        if isinstance(screens_value, list):
            seen_screens = [dict(screen) for screen in screens_value]

        sub_goals: List[Dict[str, Any]] = []
        sub_goals_value = data.get("sub_goals")
        if isinstance(sub_goals_value, list):
            sub_goals = [dict(goal) for goal in sub_goals_value if isinstance(goal, dict)]

        current_sub_goal_index_raw = data.get("current_sub_goal_index", 0)
        current_sub_goal_index = (
            int(cast("int", current_sub_goal_index_raw))
            if isinstance(current_sub_goal_index_raw, (int, float))
            else 0
        )

        state.__restore_from_data(
            sub_goals=sub_goals,
            step_count=step_count,
            is_complete=is_complete,
            seen_screens=seen_screens,
            low_delta_streak=low_delta_streak,
            last_delta_score=last_delta_score,
            completion_reason=completion_reason,
            realignment_count=realignment_count,
            current_sub_goal_index=current_sub_goal_index,
        )

        return state
