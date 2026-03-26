from __future__ import annotations

import hashlib
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, cast

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.core.services.vision import SubGoalContext, VisionService
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult, PlanResult
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
        *,
        min_confidence: float = 0.4,
    ) -> None:
        self.__vision = vision_tool
        self.__min_confidence = min_confidence

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
        elements: Optional[Dict[str, Any]] = None,
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
            delta_context=state.get_delta_context(),
            visual_hash=self.__compute_simple_hash(capture=capture),
            failures=cast("List[str]", state.build_context().get("relevant_failures", [])),
            prior_rejection_history=state.rejection_history,
        )
        # Clear rejection history after successful analysis — prevents stale
        # multi-turn context from leaking into unrelated future steps.
        state.clear_rejection_history()
        state.update_delta_context(analysis.gemini_delta)

        if analysis.content_exhausted:
            state.reset_loop_detector()
            # Do not mark_complete here: content_exhausted means "no more content on this list/feed",
            # not "task done". Marking complete would cause early exits; fall through and plan next.

        # Select and gate the action BEFORE evaluating sub-goal completion.
        # If the action is rejected (low confidence, repeated failure), we must not
        # advance the sub-goal — the LLM's completion signal is tied to an action
        # that won't execute.
        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        # Confidence gating: reject actions below the minimum confidence threshold.
        if action.confidence < self.__min_confidence:
            logger.warning(
                "[Planner] Action '%s' rejected: confidence %.2f < threshold %.2f",
                action.to_description()[:60],
                action.confidence,
                self.__min_confidence,
            )
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason=f"Action confidence {action.confidence:.2f} below threshold {self.__min_confidence}",
            )

        # Check if this EXACT action just failed on this screen hash
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
            ActionType.ASK_USER,
            ActionType.WAIT,
        }:
            state.record_sub_goal_action()

        return self.__build_plan_result(
            action=action,
            capture=capture,
            metrics=analysis.metrics,
            memories=analysis.memories,
            step_number=state.step_count,
            metadata={
                **(analysis.metadata or {}),
                "observation": analysis.screen_description,
                # Pass analysis to RECORD node for post-execution sub-goal completion check.
                "_analysis": analysis,
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

    def __build_plan_result(
        self,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        *,
        memories: int = 0,
        is_recovery: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
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
