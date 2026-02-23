from __future__ import annotations

import hashlib
from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from fathom.constants import ActionType
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
        screen_width: int,
        screen_height: int,
        *,
        use_xml: bool = False,
        prompt_if_stuck: bool = False,
        interactive_mode: bool = False,
        elements: Optional[Dict[str, Any]] = None,
    ) -> PlanResult:
        """
        Plan the next step based on current state.
        """

        logger.debug(
            f"[H5] Entering planner plan_step | "
            f"step_count={state.step_count} is_complete={state.is_complete} "
            f"is_stuck={state.is_stuck} can_continue={state.can_continue}"
        )

        if not state.can_continue:
            if state.is_complete:
                return PlanResult(step=None, is_complete=True, reason="Intent completed")

            # NATIVE INTERCEPT: Autonomous recovery exhausted, yield to HITL if enabled
            if state.is_stuck and interactive_mode and prompt_if_stuck:
                logger.warning(
                    "Agent exhausted recovery attempts. Yielding to native HITL intercept."
                )
                return PlanResult(
                    step=self.__build_step(
                        action=Action(
                            confidence=1.0,
                            target="ask_user",
                            action_type=ActionType.ASK_USER,
                            rationale="Native intercept: loop recovery exhausted.",
                            text="I am stuck in a loop and cannot recover autonomously. What should I do next?",
                        ),
                        capture=capture,
                        is_recovery=True,
                        step_number=state.step_count,
                    ),
                    is_complete=False,
                    reason="Native intercept for HITL due to exhausted recovery.",
                )

            return PlanResult(
                step=None, is_complete=False, reason="Max steps or recovery exhausted"
            )

        # Default tracking note
        current_tracking_note = state.tracking_note

        # IMMEDIATE RECOVERY
        if state.is_stuck:
            completion_error = None
            completion_signal = False
            logger.warning("Agent is stuck in a loop. Evaluating recovery options.")

            # NATIVE INTERCEPT: Yield to HITL if enabled before attempting aggressive auto-recovery
            if interactive_mode and prompt_if_stuck:
                logger.warning("Agent is stuck. Yielding to native HITL intercept.")
                return PlanResult(
                    step=self.__build_step(
                        action=Action(
                            confidence=1.0,
                            target="ask_user",
                            action_type=ActionType.ASK_USER,
                            rationale="Loop detected (Screen repeating). Requesting human intervention.",
                            text="I have detected a loop and I'm repeating the same screen state. How should I proceed to break this cycle?",
                        ),
                        capture=capture,
                        is_recovery=True,
                        step_number=state.step_count,
                    ),
                    is_complete=False,
                    reason="HITL intercept for loop detection.",
                )

            try:
                completion_signal = await self.__vision.check_completion(
                    capture=capture,
                    intent=state.intent,
                    screen_width=screen_width,
                    screen_height=screen_height,
                    context_manager=context_manager,
                    tracking_note=state.tracking_note,
                )
            except Exception as exception:
                completion_error = str(exception)

            logger.debug(
                f"[H3] Stuck gate reached | "
                f"can_recover={state.can_continue} "
                f"check_completion_signal={completion_signal} "
                f"check_completion_error={completion_error}"
            )

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
            failures=cast("List[str]", state.build_context().get("relevant_failures", [])),
        )

        completion = reasoner.analyze_completion(
            analysis=analysis, screen_description=capture.activity
        )

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        if completion.is_complete:
            state.mark_complete(reason=completion.evidence)

            # If there's a valid physical action, we should execute it before finishing
            if action.action_type not in ("complete", "unknown", "wait"):
                step = self.__build_step(
                    action=action, step_number=state.step_count, capture=capture
                )
            else:
                step = None

            return PlanResult(
                step=step,
                is_complete=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                reason=completion.evidence,
                memories=analysis.memories,
            )

        # Optimization: Check if this EXACT action just failed on this screen hash
        if state.should_avoid_action(action=action):
            logger.warning(msg=f"Avoiding recently failed action: {action.to_description()}")
            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason="Action recently failed on this screen",
            )

        if not reasoner.should_accept_action(
            action=action, has_failed_before=state.should_avoid_action(action=action)
        ):
            logger.warning(
                f"Action rejected by reasoner: {action.to_description()} "
                f"(confidence={action.confidence:.2f}, "
                f"has_failed_before={state.should_avoid_action(action=action)})"
            )

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
                    reason="Low confidence intercept for HITL.",
                )

            return PlanResult(
                step=None,
                is_complete=False,
                should_retry=True,
                metrics=analysis.metrics,
                metadata=analysis.metadata,
                memories=analysis.memories,
                reason=f"Action rejected: {action.rationale}",
            )

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
            action=action, step_number=step_number, capture=capture, is_recovery=is_recovery
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
    ) -> Step:
        """
        Helper to construct a Step object.
        """

        screen_hash: str = self.__compute_simple_hash(capture=capture)

        return Step(
            action=action,
            screen_hash=screen_hash,
            step_number=step_number,
            is_conditional=is_recovery,
            condition="recovery" if is_recovery else None,
        )

    def __compute_simple_hash(self, capture: ScreenCapture) -> str:
        """
        Compute a simple hash of the screen capture
        """

        data: bytes = f"{capture.activity}:{len(capture.image)}".encode()
        return hashlib.md5(string=data, usedforsecurity=False).hexdigest()[:16]
