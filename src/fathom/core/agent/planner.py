from __future__ import annotations

import hashlib
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, cast

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason, PlanMetadataKey
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.core.services.vision import SubGoalContext, VisionService
from fathom.schemas.actions import Action
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.results import AnalysisOutcome, AnalysisResult, PlanResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step

logger = getLogger(name=__name__)


class StepPlanner:
    """
    Plans and prepares steps for execution.
    """

    def __init__(
        self,
        vision_tool: VisionService,
    ) -> None:
        self.__vision = vision_tool

    @property
    def vision_tool(self) -> VisionService:
        """
        Returns the underlying vision tool.
        """

        return self.__vision

    async def plan_step(
        self,
        state: AgentState,
        reasoner: Reasoner,
        capture: ScreenCapture,
        context_manager: ContextManager,
        *,
        screen_width: int,
        screen_height: int,
        use_xml: bool = True,
        prompt_if_stuck: bool = False,
        interactive_mode: bool = False,
        strict_mode: bool = False,
        elements: Optional[Dict[str, Any]] = None,
        screen_observation: Optional[ScreenObservation] = None,
        last_block_reason: Optional[str] = None,
        last_block_message: Optional[str] = None,
    ) -> PlanResult:
        """
        Plan the next step based on current state.
        """

        if not state.can_continue:
            if state.is_complete:
                return PlanResult(
                    step=None, is_complete=True, reason=CompletionReason.SUCCESS.value
                )

            # TERMINAL FAIL: If we reach here, it means max_steps or budgets are hit.
            # We must stop immediately to avoid infinite HITL loops.
            return PlanResult(
                step=None,
                is_complete=True,
                reason=CompletionReason.STUCK.value
                if state.is_stuck
                else CompletionReason.FAILED.value,
            )

        # Default tracking note
        current_tracking_note = state.tracking_note

        # IMMEDIATE RECOVERY
        if state.is_stuck:
            # NATIVE INTERCEPT: Yield to HITL if enabled before attempting aggressive auto-recovery
            if interactive_mode and prompt_if_stuck:
                # If we have guidance, proceed to analysis.
                if context_manager.get_user_guidance():
                    # Pass through to analysis
                    pass
                else:
                    return PlanResult(
                        step=self.__build_step(
                            action=Action(
                                confidence=1.0,
                                target="Request user assistance",
                                action_type=ActionType.ASK_USER,
                                rationale="Loop detected (Screen repeating). Requesting human intervention.",
                                text="I have detected a loop and I'm repeating the same screen state. How should I proceed to break this cycle?",
                            ),
                            capture=capture,
                            is_recovery=True,
                            step_number=state.step_count,
                        ),
                        is_complete=False,
                        reason=CompletionReason.INTERVENTION_REQUIRED.value,
                    )

            # Autonomous Recovery (ONLY for non-interactive mode)
            else:
                recovery_action = state.get_recovery_action()
                if recovery_action:
                    return self.__build_plan_result(
                        capture=capture,
                        is_recovery=True,
                        action=recovery_action,
                        step_number=state.step_count,
                    )

        # Build minimal sub-goal context for vision (avoid passing AgentState into VisionService)
        sub_goal_info: Optional[SubGoalContext] = None
        current_sub_goal = state.get_current_sub_goal()
        if current_sub_goal and state.has_sub_goals():
            current_idx, total = state.get_sub_goal_progress()
            sub_goal_info = {
                "index": current_idx,
                "total": total,
                "description": current_sub_goal.description,
                "strict_mode": strict_mode,
                "required_action_family": (
                    current_sub_goal.execution_contract.required_action_family.value
                ),
                "scroll_axis": current_sub_goal.execution_contract.scroll_axis.value,
                "surface": current_sub_goal.execution_contract.surface or "",
            }

        analysis = await self.__vision.analyze(
            use_xml=use_xml,
            capture=capture,
            elements=elements,
            intent=state.intent,
            is_stuck=state.is_stuck,
            screen_width=screen_width,
            screen_height=screen_height,
            sub_goal_info=sub_goal_info,
            context_manager=context_manager,
            last_action=state.last_action_type,
            tracking_note=current_tracking_note,
            recent_effects=state.get_recent_effects(),
            loop_observation=state.build_loop_observation(),
            screen_observation=screen_observation,
            prior_rejection_history=state.rejection_history,
            visual_hash=self.__compute_simple_hash(capture=capture),
            failures=cast("List[str]", state.build_context().get("relevant_failures", [])),
            last_block_reason=last_block_reason,
            last_block_message=last_block_message,
        )

        # Use-once signals: rejection history, verifier feedback, and
        # user guidance are consumed by this ANALYZE iteration and then
        # cleared so a single HITL nudge cannot become a sticky
        # imperative across later iterations.
        state.clear_rejection_history()
        context_manager.clear_user_guidance()
        context_manager.clear_verifier_feedback()

        if analysis.content_exhausted:
            state.reset_loop_detector()
            # Do not mark_complete here: content_exhausted means "no more content on this list/feed",
            # not "task done". Marking complete would cause early exits; fall through and plan next.

        if analysis.outcome in (
            AnalysisOutcome.REQUEST_REPLAN,
            AnalysisOutcome.REPORT_UNACTIONABLE,
        ):
            return self.__build_escape_plan_result(
                capture=capture,
                analysis=analysis,
                step_number=state.step_count,
            )

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        if state.should_avoid_action(action=action):
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason=CompletionReason.FAILED.value,
            )

        # Check if same tap/type action repeated 3+ times on the same screen.
        # Swipes/scrolls are excluded since they legitimately repeat.
        if state.is_action_repeating_on_screen(action=action):
            repeated_desc = action.to_description()
            logger.warning(
                "[Planner] Action '%s' repeated 3+ times on same screen — forcing replan.",
                repeated_desc[:60],
            )
            # Record as failure so it appears in relevant_failures context for the LLM,
            # preventing the model from proposing the same ineffective action again.
            state.record_repeated_action_failure(action=action)

            # Store the rejected analysis metadata so vision.analyze() can build
            # a multi-turn conversation history on the next GROUND→ANALYZE cycle.
            # The LLM will see its own rejected tool call as a prior model turn.
            state.set_rejection_history(
                self.__vision.build_rejection_history_from_analysis(
                    analysis=analysis,
                    rejection_reason=(
                        f"REJECTED: '{repeated_desc}' was repeated 3+ times on the same screen "
                        "without progress. You MUST choose a completely different action."
                    ),
                )
            )

            # Signal the blocked action via PlanResult so the graph node can
            # handle guidance injection (planner should not mutate context state).
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                memories=analysis.memories,
                reason=CompletionReason.ACTION_BLOCKED.value,
                metadata={
                    **(analysis.metadata or {}),
                    "blocked_action": repeated_desc,
                },
            )

        # ── Intent completion check (non-sub-goal path only) ──
        # When sub-goals are defined, completion is driven entirely by the RECORD node
        # after each action executes. The planner must not short-circuit here — doing so
        # would return step=None and the proposed action would never reach EXECUTE.
        if not state.has_sub_goals():
            current_sub_goal = state.get_current_sub_goal()
            current_sub_goal_text = current_sub_goal.description if current_sub_goal else None
            completion = reasoner.analyze_completion(
                analysis=analysis,
                screen_description=capture.activity,
                current_sub_goal=current_sub_goal_text,
            )

            if completion.is_complete:
                state.mark_complete(reason=completion.evidence)

                return PlanResult(
                    step=None,
                    is_complete=True,
                    metrics=analysis.metrics,
                    reason=analysis.reasoning,
                    metadata=analysis.metadata,
                    memories=analysis.memories,
                )

        # Record action for sub-goal trace verification
        if state.has_sub_goals() and action.action_type not in {
            ActionType.WAIT,
            ActionType.ASK_USER,
        }:
            state.record_sub_goal_action()

        step_metadata: Dict[str, Any] = {}
        if action.action_type is ActionType.VALIDATE and (
            analysis.is_sub_goal_complete or analysis.is_goal_complete
        ):
            step_metadata["terminal_validation_candidate"] = True

        return self.__build_plan_result(
            action=action,
            capture=capture,
            metrics=analysis.metrics,
            memories=analysis.memories,
            step_number=state.step_count,
            step_metadata=step_metadata,
            metadata={
                **(analysis.metadata or {}),
                PlanMetadataKey.OBSERVATION.value: analysis.screen_description,
                # Pass analysis to RECORD node for post-execution sub-goal completion check.
                PlanMetadataKey.ANALYSIS.value: analysis,
            },
        )

    def __select_action(
        self,
        state: AgentState,
        reasoner: Reasoner,
        analysis: AnalysisResult,
    ) -> Action:
        """
        Return the best action from the analysis result.
        """

        context = state.build_context()
        failures_raw = context.get("relevant_failures", [])
        failures = failures_raw if isinstance(failures_raw, list) else []

        return reasoner.select_best_action(
            primary=analysis.action,
            alternatives=analysis.alternatives,
            failed_actions={str(failure) for failure in failures},
        )

    def __build_escape_plan_result(
        self,
        *,
        step_number: int,
        capture: ScreenCapture,
        analysis: AnalysisResult,
    ) -> PlanResult:
        """
        Translate a ``REQUEST_REPLAN`` analysis into a PlanResult routed
        by the structured :class:`EscapeReport` category.

        Replan categories surface as a step-less PlanResult with
        ``reason=CompletionReason.REQUEST_REPLAN.value`` so the graph
        node dispatches the recovery coordinator with the escape report
        attached. Human categories surface as an ``ASK_USER`` step whose
        ``text`` is the escape detail so EXECUTE escalates through the
        existing HITL path. The placeholder WAIT action on the analysis
        never reaches EXECUTE in either branch.

        ``step_number`` is the caller's current ``AgentState.step_count``
        and is threaded through to the ASK_USER ``Step`` so telemetry
        and history attribute the human-escalation to the right step.
        """

        if (escape_report := analysis.escape_report) is None:
            raise ValueError(
                "AnalysisResult.outcome=REQUEST_REPLAN requires a populated escape_report"
            )

        if escape_report.routes_to_replan():
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                memories=analysis.memories,
                reason=CompletionReason.REQUEST_REPLAN.value,
                metadata={
                    **(analysis.metadata or {}),
                    PlanMetadataKey.ESCAPE_REPORT.value: escape_report.model_dump(mode="json"),
                },
            )

        ask_user_action = Action(
            confidence=1.0,
            text=escape_report.detail,
            rationale=escape_report.detail,
            target="Request user assistance",
            action_type=ActionType.ASK_USER,
        )

        return PlanResult(
            step=self.__build_step(
                capture=capture,
                is_recovery=True,
                action=ask_user_action,
                step_number=step_number,
            ),
            is_complete=False,
            metrics=analysis.metrics,
            memories=analysis.memories,
            reason=CompletionReason.INTERVENTION_REQUIRED.value,
            metadata={
                **(analysis.metadata or {}),
                PlanMetadataKey.ESCAPE_REPORT.value: escape_report.model_dump(mode="json"),
            },
        )

    def __build_plan_result(
        self,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        *,
        memories: int = 0,
        is_recovery: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        step_metadata: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> PlanResult:
        """
        Return a PlanResult with the given action and metadata.
        """

        step: Step = self.__build_step(
            action=action,
            capture=capture,
            is_recovery=is_recovery,
            step_number=step_number,
            event_type=(metadata or {}).get("event_type"),
            metadata=step_metadata,
        )

        return PlanResult(
            step=step,
            is_complete=False,
            memories=memories,
            metrics=metrics or {},
            metadata=metadata or {},
            is_valid_action=action.is_valid,
            validation_reasoning=action.validation_reason,
            reason=action.rationale or ("Step planned" if not is_recovery else "Recovery step"),
        )

    def __build_step(
        self,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        is_recovery: bool = False,
        event_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Step:
        """
        Helper to construct a Step object.
        """

        screen_hash: str = self.__compute_simple_hash(capture=capture)

        validated_event_type: Optional[Literal["action", "validation"]] = None

        if event_type == "action":
            validated_event_type = "action"

        elif event_type == "validation":
            validated_event_type = "validation"

        return Step(
            action=action,
            screen_hash=screen_hash,
            step_number=step_number,
            is_conditional=is_recovery,
            event_type=validated_event_type,
            condition="recovery" if is_recovery else None,
            metadata=metadata or {},
        )

    def __compute_simple_hash(self, *, capture: ScreenCapture) -> str:
        """
        Return a stable screen identity for the current capture.
        """

        if capture.state is not None and capture.state.visual_hash:
            return capture.state.visual_hash[:16]

        logger.warning("Screen capture state missing visual_hash; falling back to activity hash")

        return hashlib.md5(
            capture.activity.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:16]
