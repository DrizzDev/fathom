from __future__ import annotations

import hashlib
from logging import getLogger
from typing import Any, Dict, Optional

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
        use_xml: bool = False,
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

            return PlanResult(
                step=None, is_complete=False, reason="Max steps or recovery exhausted"
            )

        # IMMEDIATE RECOVERY
        if state.is_stuck:
            completion_error = None
            completion_signal = False
            logger.warning("Agent is stuck in a loop. Forcing recovery action.")

            try:
                completion_signal = await self.__vision.check_completion(
                    intent=state.intent,
                    capture=capture,
                    context_manager=context_manager,
                    agent_state=state,
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

        from typing import List, cast

        analysis = await self.__vision.analyze(
            use_xml=use_xml,
            capture=capture,
            elements=elements,
            intent=state.intent,
            context_manager=context_manager,
            agent_state=state,
            failures=cast("List[str]", state.build_context().get("relevant_failures", [])),
        )

        completion = reasoner.analyze_completion(
            analysis=analysis, screen_description=capture.activity
        )

        action = self.__select_action(state=state, reasoner=reasoner, analysis=analysis)

        if completion.is_complete:
            state.mark_complete(reason=completion.evidence)

            # If there's a valid physical action, we should execute it before finishing
            step = None
            if action.action_type not in ("complete", "unknown", "wait"):
                step = self.__build_step(
                    action=action, step_number=state.step_count, capture=capture
                )

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
            metadata=analysis.metadata,
            memories=analysis.memories,
            step_number=state.step_count,
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
            failed_actions={str(object=failure) for failure in failures},
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
            reason=action.rationale or ("Step planned" if not is_recovery else "Recovery step"),
            is_valid_action=action.is_valid,
            validation_reasoning=action.validation_reason,
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
