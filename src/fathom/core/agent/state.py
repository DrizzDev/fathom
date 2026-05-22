from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, cast

from fathom.constants import ActionExecutionKind, ActionType
from fathom.constants.runtime import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_LOOP_THRESHOLD,
    DEFAULT_MAX_STEPS,
    DEFAULT_REALIGNMENT_BUDGET,
)
from fathom.constants.screen import (
    LOOP_OSCILLATION_AB_WINDOW,
    MIN_LOOP_OBSERVATION_REPETITIONS,
    MIN_NO_PROGRESS_FOR_OBSERVATION,
    MIN_SCREENS_FOR_NEAR_DUPLICATE,
)
from fathom.constants.state import CompletionReason
from fathom.core.recovery.ladder import RecoveryActionLadder
from fathom.core.runtime import ExecutionTaskAdapter, RuntimeState, TargetIdentity
from fathom.schemas.actions import Action
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.effect import ActionEffect
from fathom.schemas.observation import LoopObservation, ScreenRelation
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.screens import ScreenState
from fathom.schemas.state import (
    ActionHistory,
    InteractionTracker,
    LoopDetectorState,
)
from fathom.schemas.steps import StepResult
from fathom.schemas.subgoal import SubGoal, SubGoalStatus
from fathom.schemas.supervision import BlockReason
from fathom.schemas.tasks import ExecutionTaskState

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
        max_steps: int = DEFAULT_MAX_STEPS,
        loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        realignment_budget: int = DEFAULT_REALIGNMENT_BUDGET,
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

        self.__interaction_tracker = InteractionTracker()
        self.__action_history = ActionHistory(max_size=context_window)

        self.__runtime = RuntimeState.create(
            loop_threshold=loop_threshold,
            loop_window=context_window,
            realignment_budget=realignment_budget,
        )
        self.__recovery_ladder = RecoveryActionLadder()

        self.__is_complete = False
        self.__completion_reason: Optional[str] = None

        # Enhanced state fields
        self.__knowledge: Dict[str, Any] = {}
        self.__current_screen_name: Optional[str] = None

        self.__last_error: Optional[str] = None
        self.__last_action_type: Optional[str] = None
        self.__last_action_description: Optional[str] = None

        # Legacy delta-context fields. Kept as immutable defaults so the
        # checkpoint round-trip continues to accept payloads written by
        # earlier versions of the schema. No code path mutates them
        # any more — the signal lives in runtime effect history instead.
        self.__low_delta_streak: int = 0
        self.__last_delta_score: Optional[float] = None

        # Multi-turn rejection history for cross-iteration feedback loops.
        # Stores provider-neutral ConversationTurn objects so the next
        # vision.analyze() call can pass them as conversation_history.
        self.__rejection_history: Optional[List[ConversationTurn]] = None

        # Sub-goal tracking for sequential intent execution.
        # The runtime aggregate owns task progress; these fields remain as
        # compatibility shims while existing consumers migrate to runtime.tasks.
        self.__sub_goals: List[SubGoal] = []
        self.__current_sub_goal_index: int = 0
        self.__sub_goal_action_count: int = 0
        self.__sub_goal_start_screen: Optional[str] = None

        # Counts how many consecutive ANALYZE turns produced
        # ``is_complete=True`` while sub-goals were still open and the
        # router deferred to GROUND. Bounded retry prevents the planner
        # from looping on a "complete" verdict the runtime cannot honour.
        self.__consecutive_complete_deferrals: int = 0

    @property
    def intent(self) -> str:
        """
        The goal being pursued.
        """

        return self.__intent

    @property
    def runtime(self) -> RuntimeState:
        """
        Return decomposed runtime state components.
        """

        return self.__runtime

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

        return self.__runtime.screen.detector.is_stuck()

    @property
    def last_delta_score(self) -> Optional[float]:
        """
        Most recent post-action screen-change magnitude in [0.0, 1.0].
        """

        return self.__last_delta_score

    def bump_realignment_budget(self) -> None:
        """
        Increment the realignment counter so :attr:`can_continue`
        enforces the per-run intervention budget.
        """

        self.__runtime.realignment.record()

    def reset_loop_history(self) -> None:
        """
        Drop accumulated loop-detection evidence (e.g. after a user
        intervention course-corrects from the stuck path).
        """

        self.__runtime.screen.detector.reset()

    def record_hitl_intervention(self) -> None:
        """
        Composite of :meth:`bump_realignment_budget` and
        :meth:`reset_loop_history`. Use the granular methods when only
        one effect is desired.
        """

        self.bump_realignment_budget()
        self.reset_loop_history()

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
            if self.__runtime.realignment.count >= self.__runtime.realignment.budget:
                return False

            # Autonomous budget (used in non-interactive mode)
            return bool(self.__runtime.screen.detector.can_recover())

        return True

    @property
    def current_screen(self) -> Optional[ScreenState]:
        """
        Current screen state.
        """

        return self.__runtime.screen.current

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

        return self.__runtime.screen.is_new(screen=screen)

    def update_screen(self, screen: ScreenState) -> bool:
        """
        Update current screen state.

        Args:
            screen: New screen state.

        Returns:
            True if this is a new screen, False if seen before.
        """

        previous_screen = self.__runtime.screen.current
        is_new_screen = self.__runtime.screen.is_new(screen=screen)
        self.__runtime.screen.update(screen=screen, observation=None)

        if is_new_screen:
            self.__runtime.screen.remember(screen=screen)
            logger.debug(f"New screen detected: {screen.visual_hash[:8]} ({screen.activity})")
            # Loop detector owns the visual-progress policy:
            # it advances only when the new screen is visually distinct from the previous one, ignoring xml/interaction hash flips.
            self.__runtime.screen.detector.observe_screen(previous=previous_screen, current=screen)

        elif previous_screen and previous_screen.activity_hash != screen.activity_hash:
            logger.debug(
                f"Activity changed on known screen: {previous_screen.activity} -> "
                f"{screen.activity}. Preserving loop detector state."
            )
        else:
            logger.debug(f"Returning to known screen: {screen.visual_hash[:8]}")

        last_effect = self.__runtime.effects.last_effect()

        effect_status = (
            last_effect.status
            if self.__last_action_type is not None and last_effect is not None
            else None
        )
        self.__runtime.screen.detector.record(
            screen=screen,
            action_type=self.__last_action_type,
            action_description=self.__last_action_description,
            effect_status=effect_status,
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
        activity = (
            self.__runtime.screen.current.activity if self.__runtime.screen.current else "unknown"
        )
        self.__action_history.record_action(
            action=result.step.action, success=result.success, activity=activity
        )

        if result.step.action.execution_kind is ActionExecutionKind.DEVICE:
            self.__interaction_tracker.record(action_type=result.step.action.action_type.value)
            self.__last_action_type = result.step.action.action_type.value
            self.__last_action_description = result.step.action.to_description()
        else:
            self.__last_action_type = None
            self.__last_action_description = None

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

    @property
    def consecutive_complete_deferrals(self) -> int:
        """
        Number of consecutive ANALYZE turns whose ``is_complete=True``
        verdict the router deferred because sub-goals were still open.
        """

        return self.__consecutive_complete_deferrals

    def record_complete_deferral(self) -> int:
        """
        Increment and return the consecutive complete-deferral counter.

        The router invokes this when it routes an ``is_complete=True``
        ANALYZE verdict back to GROUND because sub-goals are still
        open. Returning the post-increment value lets the caller
        decide whether to escalate.
        """

        self.__consecutive_complete_deferrals += 1
        return self.__consecutive_complete_deferrals

    def reset_complete_deferrals(self) -> None:
        """
        Zero the consecutive complete-deferral counter.

        Called on real progress: a non-complete plan, a successful
        sub-goal advancement, or a verifier pass. Without the reset,
        a single late deferral would forever bias the bounded-retry
        threshold against the planner.
        """

        self.__consecutive_complete_deferrals = 0

    def set_sub_goals(self, sub_goals: List[SubGoal]) -> None:
        """
        Set the decomposed sub-goals for this intent.

        Args:
            sub_goals: List of sequential sub-goals to execute.
        """

        self.__sub_goals = [goal.model_copy(deep=True) for goal in sub_goals]
        self.__current_sub_goal_index = 0

        self.__runtime.tasks.load(
            tasks=ExecutionTaskAdapter().from_sub_goals(sub_goals=self.__sub_goals),
        )

        if self.__sub_goals:
            self.__sub_goals[0].mark_in_progress()
            # Track starting screen for first sub-goal
            if self.__runtime.screen.current:
                self.__sub_goal_start_screen = self.__runtime.screen.current.visual_hash

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
            trace_verified=False,
            evidence=completion_signal.evidence,
            keyword_match=completion_signal.keyword_match,
            llm_confidence=completion_signal.llm_confidence,
            action_executed=completion_signal.action_executed,
            flagged_complete=completion_signal.flagged_complete,
            rationale_verified=completion_signal.rationale_verified,
        )

        # Mark complete with all signals
        current.mark_complete(
            trace_verified=updated_signal.trace_verified,
            flagged_complete=updated_signal.flagged_complete,
            rationale_verified=updated_signal.rationale_verified,
        )

        # Sub-goal advanced -> any in-flight complete-deferral streak is now
        # stale; the next planner verdict starts from a clean slate.
        self.__consecutive_complete_deferrals = 0

        logger.info(
            "[AgentState] Sub-goal marked complete",
            extra={
                "component": "agent_state",
                "event": "subgoal_complete",
                "sub_goal_index": current.index,
                "evidence": updated_signal.evidence,
                "sub_goal_description": current.description[:80],
                "claim_verified": updated_signal.claim_verified,
                "trace_verified": updated_signal.trace_verified,
                "action_effective": updated_signal.action_effective,
            },
        )

        # Advance to next sub-goal
        self.__current_sub_goal_index += 1
        self.__runtime.tasks.mark(state=ExecutionTaskState.SUCCEEDED)
        self.__runtime.tasks.advance()

        if self.__current_sub_goal_index < len(self.__sub_goals):
            next_goal = self.__sub_goals[self.__current_sub_goal_index]
            next_goal.mark_in_progress()

            # Reset per-sub-goal counters on advancement.
            if self.__runtime.screen.current:
                self.__sub_goal_start_screen = self.__runtime.screen.current.visual_hash

            self.__sub_goal_action_count = 0
            logger.info(
                f"[AgentState] Advanced to sub-goal {next_goal.index}: {next_goal.description}"
            )
            return True
        else:
            logger.info("[AgentState] All sub-goals complete")
            return False

    def reopen_last_completed_sub_goal(self) -> bool:
        """
        Re-activate the most recently completed sub-goal after a verifier rejection.

        Returns ``True`` when a completed terminal sub-goal was restored as the active
        mission, otherwise ``False``.
        """

        if self.get_current_sub_goal() is not None or not self.__sub_goals:
            return False

        last_index = len(self.__sub_goals) - 1
        if last_index < 0:
            return False

        candidate = self.__sub_goals[last_index]
        if not candidate.is_complete():
            return False

        candidate.status = SubGoalStatus.IN_PROGRESS
        candidate.flagged_complete = False
        candidate.trace_verified = False
        candidate.rationale_verified = False
        candidate.completion_verified = False
        self.__current_sub_goal_index = last_index
        self.__is_complete = False
        self.__completion_reason = None

        self.__runtime.tasks.load(
            tasks=ExecutionTaskAdapter().from_sub_goals(sub_goals=self.__sub_goals),
        )
        for _ in range(self.__current_sub_goal_index):
            self.__runtime.tasks.advance()

        logger.info(
            "[AgentState] Reopened last completed sub-goal after verifier rejection",
            extra={
                "component": "agent_state",
                "event": "subgoal_reopened",
                "sub_goal_index": candidate.index,
                "sub_goal_description": candidate.description[:80],
            },
        )
        return True

    def record_sub_goal_action(self) -> None:
        """
        Record that an action was executed for the current sub-goal.
        Call this from planner after executing each action.
        """

        self.__sub_goal_action_count += 1
        self.__runtime.tasks.record_attempt()

    @property
    def current_sub_goal_action_count(self) -> int:
        """
        Number of actions executed against the active sub-goal.
        Reset to zero on sub-goal advance and on replan.
        """

        return self.__sub_goal_action_count

    @property
    def current_sub_goal_over_budget(self) -> bool:
        """
        Whether the active sub-goal has consumed at least
        :attr:`SubGoal.max_steps` actions without advancing.

        Returns ``False`` when there is no active sub-goal (between
        runs, after completion). The RECORD node uses this property to
        decide whether to escalate via ``SUBGOAL_BUDGET_EXCEEDED``.
        """

        current = self.get_current_sub_goal()
        if current is None:
            return False

        return self.__sub_goal_action_count >= current.max_steps

    def replan_pending_sub_goals(self, *, new_sub_goals: List[SubGoal]) -> None:
        """
        Replace every unfinished sub-goal with ``new_sub_goals``
        (preserving completed work) and mark the first new sub-goal IN_PROGRESS.

        Preserves the per-sub-goal action counter when the new first sub-goal names the same target as the previously active sub-goal.
        A cosmetic replan that does not change the imperative the agent is attempting. The counter must accumulate across cosmetic replans
        so ``SUBGOAL_BUDGET_EXCEEDED`` can fire when the same target is retried repeatedly under ``TARGET_UNRESOLVED`` (or any other repeat-triggering signal).
        """

        previous_active = self.get_current_sub_goal()
        completed = [goal for goal in self.__sub_goals if goal.is_complete()]

        reindexed = [
            goal.model_copy(update={"index": len(completed) + offset})
            for offset, goal in enumerate(new_sub_goals)
        ]

        self.__sub_goals = completed + reindexed
        self.__current_sub_goal_index = len(completed)
        self.__runtime.tasks.load(
            tasks=ExecutionTaskAdapter().from_sub_goals(sub_goals=self.__sub_goals),
        )
        for _ in range(self.__current_sub_goal_index):
            self.__runtime.tasks.advance()

        same_target = (
            bool(reindexed)
            and previous_active is not None
            and TargetIdentity.describes_same_target(
                previous=previous_active.description,
                replacement=reindexed[0].description,
            )
        )
        if not same_target:
            self.__sub_goal_action_count = 0

        if self.__current_sub_goal_index < len(self.__sub_goals):
            current = self.__sub_goals[self.__current_sub_goal_index]
            current.mark_in_progress()
            logger.info(
                "[AgentState] Replanned",
                extra={
                    "event": "replanned",
                    "kept": len(completed),
                    "component": "agent.state",
                    "replaced": len(reindexed),
                    "counter_preserved": same_target,
                    "current": current.description[:60],
                    "sub_goal_action_count": self.__sub_goal_action_count,
                },
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

    def get_sub_goal_progress(self) -> Tuple[int, int]:
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
        Get the next mechanical recovery action when the agent is stuck.
        """

        return self.__recovery_ladder.next(detector=self.__runtime.screen.detector)

    def record_recovery_attempt(self) -> int:
        """
        Record a planner-level recovery attempt.
        """

        return self.__runtime.screen.detector.record_recovery_attempt()

    def reset_loop_detector(self) -> None:
        """
        Reset loop detector state after an explicit progress signal.
        """

        self.__runtime.screen.detector.signal_content_exhausted()

    def record_action_effect(self, *, effect: ActionEffect) -> None:
        """
        Append a structured action-effect outcome to the rolling
        trajectory window.

        Called from EXECUTE after each step using
        :meth:`ActionEffect.from_screen_diff` against the post-action
        :class:`ScreenDiff`. The window is bounded by
        ``ACTION_EFFECT_TRAJECTORY_WINDOW`` so the prompt size stays
        stable regardless of run length.
        """

        self.__runtime.effects.record_effect(effect=effect)

    def get_recent_effects(self) -> List[ActionEffect]:
        """
        Return the rolling window of recent action effects (oldest first).
        """

        return self.__runtime.effects.recent_effects()

    def get_last_action_effect(self) -> Optional[ActionEffect]:
        """
        Return the most recent recorded action effect, or ``None`` when
        no action has produced a classifiable outcome yet (e.g. very
        first step of a run, before EXECUTE has fired).
        """

        return self.__runtime.effects.last_effect()

    def build_loop_observation(self) -> Optional[LoopObservation]:
        """
        Construct a :class:`LoopObservation` summarizing the current
        stuck evidence, or ``None`` when the agent is not stuck.

        The observation is the structured input the ANALYZE prompt
        renders into ``<SYSTEM_OBSERVATION>``. Built here (not in the
        prompt assembler) so the rules for *when* to inject — and what
        evidence to surface — live with the state that produced the
        evidence.

        Returns ``None`` when:

        - The loop detector hasn't fired (``is_stuck`` is False) AND
        - The action-effect trajectory hasn't crossed the no-progress
          recovery threshold.

        In either of those branches the agent receives no
        ``SYSTEM_OBSERVATION`` block — the runtime tells the agent
        nothing when there's nothing reliable to tell.
        """

        stuck = self.is_stuck
        no_progress_run = self.consecutive_no_progress_count
        if not stuck and no_progress_run < MIN_NO_PROGRESS_FOR_OBSERVATION:
            return None

        recent_actions = [
            entry.get("action") or entry.get("action_type") or "unknown"
            for entry in self.__action_history.get_history_items()
            if isinstance(entry, dict)
        ]
        if not recent_actions:
            return None

        counts: Dict[str, int] = {}
        for descriptor in recent_actions:
            counts[descriptor] = counts.get(descriptor, 0) + 1
        repeated, count = max(counts.items(), key=lambda item: item[1])
        if count < MIN_LOOP_OBSERVATION_REPETITIONS:
            return None

        progress_scores = [
            round(effect.visual_progress, 3) for effect in self.__runtime.effects.recent_effects()
        ]

        relation = self.__classify_screen_relation()

        return LoopObservation(
            count=count,
            note=None,
            repeated_action=repeated,
            screen_relation=relation,
            progress_scores=progress_scores,
        )

    def __classify_screen_relation(self) -> ScreenRelation:
        """
        Bucket the recent screen history into a coarse
        :class:`ScreenRelation`.
        """

        screen_count = len(self.__runtime.screen.seen)

        if screen_count < MIN_SCREENS_FOR_NEAR_DUPLICATE:
            return ScreenRelation.DIVERGING

        last_two = self.__runtime.screen.seen[-MIN_SCREENS_FOR_NEAR_DUPLICATE:]
        if last_two[0].is_same_screen(last_two[1]):
            return ScreenRelation.NEAR_DUPLICATE

        if screen_count >= LOOP_OSCILLATION_AB_WINDOW:
            tail = self.__runtime.screen.seen[-LOOP_OSCILLATION_AB_WINDOW:]
            if tail[0].is_same_screen(tail[2]) and tail[1].is_same_screen(tail[3]):
                return ScreenRelation.OSCILLATING

        return ScreenRelation.DIVERGING

    @property
    def consecutive_no_progress_count(self) -> int:
        """
        Number of trailing actions classified as ``NO_PROGRESS``.

        Used by the RECORD node to decide whether to emit the
        ``NO_PROGRESS`` recovery trigger. Counts only the contiguous
        tail of the trajectory window — a single ``PROGRESS`` step
        resets the counter.
        """

        return self.__runtime.effects.consecutive_no_progress()

    def build_context(self) -> Dict[str, object]:
        """
        Build context for vision-language model with token optimization.
        """

        current_activity = (
            self.__runtime.screen.current.activity if self.__runtime.screen.current else "unknown"
        )

        context: Dict[str, object] = {
            "intent": self.__intent,
            "is_stuck": self.is_stuck,
            "max_steps": self.__max_steps,
            "step_count": self.__step_count,
            "unique_screens_seen": len(self.__runtime.screen.seen),
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

        activity = (
            self.__runtime.screen.current.activity if self.__runtime.screen.current else "unknown"
        )
        self.__action_history.record_action(action=action, success=False, activity=activity)
        self.__runtime.failures.record(
            action=action,
            reason=BlockReason.REPEATED_NO_EFFECT,
            detail=(
                f"Action {action.to_description()[:80]!r} repeated 3+ times "
                f"without progress on activity {activity!r}."
            ),
        )
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

        if (current := self.__runtime.screen.current) is None:
            return False

        return self.__runtime.screen.detector.has_repeated_action_on_same_screen(
            action_description=action.to_description(),
            screen_hash=current.visual_hash,
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
            "realignment_count": self.__runtime.realignment.count,
            "completion_reason": self.__completion_reason,
            "realignment_budget": self.__runtime.realignment.budget,
            "realignment_state": self.__runtime.realignment.to_state(),
            "healing_state": self.__runtime.healing.to_state(),
            "last_delta_score": self.__last_delta_score,
            "low_delta_streak": self.__low_delta_streak,
            "seen_screens": [screen.model_dump() for screen in self.__runtime.screen.seen],
            "sub_goals": [goal.model_dump(mode="json") for goal in self.__sub_goals],
            "current_sub_goal_index": self.__current_sub_goal_index,
            "sub_goal_action_count": self.__sub_goal_action_count,
            "consecutive_complete_deferrals": self.__consecutive_complete_deferrals,
            "recent_effects": [
                effect.model_dump(mode="json") for effect in self.get_recent_effects()
            ],
            "loop_detector_state": self.__runtime.screen.detector.to_state().model_dump(
                mode="json"
            ),
            # Diagnostic snapshot retained for backward-compatibility with
            # external checkpoint readers. Not consumed during restore.
            "action_stats": self.__action_history.get_stats(),
            "action_context": self.__action_history.get_context(),
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
        loop_detector_state: Optional[Dict[str, Any]] = None,
        recent_effects: Optional[List[Dict[str, Any]]] = None,
        sub_goal_action_count: int = 0,
        consecutive_complete_deferrals: int = 0,
        realignment_state: Optional[Dict[str, Any]] = None,
        healing_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Restore internal state from checkpoint data.
        """

        self.__step_count = step_count
        self.__is_complete = is_complete
        self.__completion_reason = completion_reason
        self.__consecutive_complete_deferrals = max(0, consecutive_complete_deferrals)

        if realignment_state is not None:
            self.__runtime.realignment.load_state(state=realignment_state)
        else:
            for _ in range(max(0, realignment_count - self.__runtime.realignment.count)):
                self.__runtime.realignment.record()

        if healing_state is not None:
            self.__runtime.healing.load_state(state=healing_state)

        self.__last_delta_score = last_delta_score
        self.__low_delta_streak = max(0, low_delta_streak)
        self.__sub_goal_action_count = max(0, sub_goal_action_count)

        self.__runtime.screen.load_seen(
            screens=[ScreenState(**data) for data in seen_screens],
        )

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

        if loop_detector_state:
            self.__runtime.screen.detector.restore(
                state=LoopDetectorState.model_validate(loop_detector_state)
            )

        self.__runtime.effects.load_effects(
            effects=[ActionEffect.model_validate(effect) for effect in recent_effects or []]
        )

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
            int(cast("int", max_steps_value))
            if isinstance(max_steps_value, (int, float))
            else DEFAULT_MAX_STEPS
        )

        realignment_budget_value = data.get("realignment_budget")
        realignment_budget = (
            int(cast("int", realignment_budget_value))
            if isinstance(realignment_budget_value, (int, float))
            else DEFAULT_REALIGNMENT_BUDGET
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

        sub_goal_action_count_raw = data.get("sub_goal_action_count", 0)
        sub_goal_action_count = (
            int(cast("int", sub_goal_action_count_raw))
            if isinstance(sub_goal_action_count_raw, (int, float))
            else 0
        )

        consecutive_complete_deferrals_raw = data.get("consecutive_complete_deferrals", 0)
        consecutive_complete_deferrals = (
            int(cast("int", consecutive_complete_deferrals_raw))
            if isinstance(consecutive_complete_deferrals_raw, (int, float))
            else 0
        )

        recent_effects: List[Dict[str, Any]] = []
        recent_effects_value = data.get("recent_effects")
        if isinstance(recent_effects_value, list):
            recent_effects = [
                dict(effect) for effect in recent_effects_value if isinstance(effect, dict)
            ]

        loop_detector_state_raw = data.get("loop_detector_state")
        loop_detector_state = (
            dict(loop_detector_state_raw) if isinstance(loop_detector_state_raw, dict) else None
        )

        realignment_state_raw = data.get("realignment_state")
        realignment_state = (
            dict(realignment_state_raw) if isinstance(realignment_state_raw, dict) else None
        )

        healing_state_raw = data.get("healing_state")
        healing_state = dict(healing_state_raw) if isinstance(healing_state_raw, dict) else None

        state.__restore_from_data(
            sub_goals=sub_goals,
            step_count=step_count,
            is_complete=is_complete,
            seen_screens=seen_screens,
            low_delta_streak=low_delta_streak,
            last_delta_score=last_delta_score,
            completion_reason=completion_reason,
            realignment_count=realignment_count,
            realignment_state=realignment_state,
            healing_state=healing_state,
            current_sub_goal_index=current_sub_goal_index,
            loop_detector_state=loop_detector_state,
            recent_effects=recent_effects,
            sub_goal_action_count=sub_goal_action_count,
            consecutive_complete_deferrals=consecutive_complete_deferrals,
        )

        return state
