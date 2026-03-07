from __future__ import annotations

import hashlib
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, cast

from fathom.constants import ActionType
from fathom.constants.state import CompletionReason
from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.context.manager import ContextManager
from fathom.core.services.vision import VisionService
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

        analysis = await self.__vision.analyze(
            use_xml=use_xml,
            capture=capture,
            elements=elements,
            intent=state.intent,
            screen_width=screen_width,
            screen_height=screen_height,
            context_manager=context_manager,
            tracking_note=current_tracking_note,
            is_stuck=state.is_stuck,
            last_action=state.last_action_type,
            delta_context=state.get_delta_context(),
            failures=cast("List[str]", state.build_context().get("relevant_failures", [])),
            agent_state=state,
        )
        state.update_delta_context(analysis.gemini_delta)

        if analysis.content_exhausted:
            state.reset_loop_detector()
            state.mark_complete(reason="Content exhaustion signaled by model")
            return PlanResult(
                step=None,
                is_complete=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason="Model signaled content exhaustion (end of list/carousel).",
            )

        # Get current sub-goal for sequential intent execution
        current_sub_goal = state.get_current_sub_goal()
        current_sub_goal_text = current_sub_goal.description if current_sub_goal else None

        # Check completion using multi-signal verification for sub-goals
        if current_sub_goal and state.has_sub_goals():
            # Get progress for logging
            current_idx, total = state.get_sub_goal_progress()

            # Use dedicated sub-goal completion analysis
            sub_goal_signal = reasoner.analyze_subgoal_completion(
                analysis=analysis,
                sub_goal_description=current_sub_goal.description,
                screen_description=capture.activity,
            )

            # Determine required threshold: validation/verification steps only need 1-2 signals
            is_validation_step = any(
                keyword in current_sub_goal.description.lower()
                for keyword in ["validate", "verify", "confirm", "check if", "check that"]
            )
            required_threshold = 1 if is_validation_step else 2
            signal_count = sub_goal_signal.count_signals()

            logger.info(
                f"[StepPlanner] Sub-goal completion analysis: '{current_sub_goal.description[:50]}...' | "
                f"signals={signal_count}/{required_threshold} | "
                f"llm={sub_goal_signal.llm_signaled} | "
                f"rationale={sub_goal_signal.rationale_verified} | "
                f"action={sub_goal_signal.action_executed} | "
                f"evidence: {sub_goal_signal.evidence}"
            )

            # Completion gate with adaptive threshold
            if sub_goal_signal.meets_threshold(required_signals=required_threshold):
                # Mark current sub-goal as complete with all signals
                has_more = state.mark_current_sub_goal_complete(completion_signal=sub_goal_signal)

                if has_more:
                    # More sub-goals remain - continue execution
                    next_sub_goal = state.get_current_sub_goal()
                    logger.info(
                        f"[StepPlanner] ✓ Sub-goal {current_sub_goal.index} COMPLETE. "
                        f"Advancing to sub-goal {next_sub_goal.index if next_sub_goal else '(none)'}: "
                        f"'{next_sub_goal.description if next_sub_goal else ''}'"
                    )
                    logger.info(
                        f"[StepPlanner] Sub-goal {current_sub_goal.index} complete with "
                        f"{sub_goal_signal.count_signals()}/3 signals. "
                        f"{state.get_sub_goal_progress()[1] - state.get_sub_goal_progress()[0]} sub-goals remain."
                    )
                    # Do not execute stale action planned for previous sub-goal.
                    # Force a re-plan against the next active sub-goal.
                    return PlanResult(
                        step=None,
                        is_complete=False,
                        should_retry=True,
                        metrics=analysis.metrics,
                        memories=analysis.memories,
                        reason="Advanced to next sub-goal; replanning next action",
                        metadata={
                            **(analysis.metadata or {}),
                            "observation": analysis.screen_description,
                            "sub_goal_completed": current_sub_goal.description,
                            "completion_signals": sub_goal_signal.count_signals(),
                        },
                    )
                else:
                    # All sub-goals complete - mark intent complete
                    logger.info(
                        f"[StepPlanner] All sub-goals complete. Final sub-goal had "
                        f"{sub_goal_signal.count_signals()}/3 signals."
                    )
                    state.mark_complete(reason="All sub-goals completed sequentially")

                    return PlanResult(
                        step=None,  # No physical actions after goal completion
                        is_complete=True,
                        metrics=analysis.metrics,
                        reason="All sub-goals completed sequentially",
                        metadata=analysis.metadata,
                        memories=analysis.memories,
                    )
            else:
                # Completion gates not met - log detailed reason
                logger.warning(
                    f"[StepPlanner] Sub-goal {current_sub_goal.index} NOT completing yet: "
                    f"{signal_count}/{required_threshold} signals | "
                    f"Progress: [{current_idx + 1}/{total}] | "
                    f"Type: {'validation' if is_validation_step else 'action'} | "
                    f"Evidence: {sub_goal_signal.evidence} | "
                    f"Will retry next step..."
                )
        else:
            # No sub-goals or checking overall intent - use original completion logic
            completion = reasoner.analyze_completion(
                analysis=analysis,
                screen_description=capture.activity,
                current_sub_goal=current_sub_goal_text,
            )

            if completion.is_complete:
                state.mark_complete(reason=completion.evidence)

                return PlanResult(
                    step=None,  # No physical actions after goal completion
                    is_complete=True,
                    metrics=analysis.metrics,
                    reason=analysis.reasoning,
                    metadata=analysis.metadata,
                    memories=analysis.memories,
                )

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        # Optimization: Check if this EXACT action just failed on this screen hash
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

        if not reasoner.should_accept_action(
            action=action, has_failed_before=state.should_avoid_action(action=action)
        ):
            # If interactive, yield to user for guidance on low confidence
            if interactive_mode and prompt_if_stuck:
                return PlanResult(
                    step=self.__build_step(
                        action=Action(
                            confidence=1.0,
                            target="ask_user",
                            action_type=ActionType.ASK_USER,
                            rationale=f"Confidence low ({action.confidence:.2f}): {action.rationale}",
                            text=f"I'm not sure what to do next. I thought about: {action.to_description()}, but my confidence is low. How should I proceed?",
                        ),
                        capture=capture,
                        is_recovery=True,
                        step_number=state.step_count,
                    ),
                    is_complete=False,
                    reason=CompletionReason.INTERVENTION_REQUIRED.value,
                )

            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason=CompletionReason.FAILED.value,
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

        validated_event_type: Literal["action", "validation"] | None = None

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

    def __compute_simple_hash(self, capture: ScreenCapture) -> str:
        """
        Compute a simple hash of the screen capture
        """

        data: bytes = f"{capture.activity}:{len(capture.image)}".encode()
        return hashlib.md5(string=data, usedforsecurity=False).hexdigest()[:16]
