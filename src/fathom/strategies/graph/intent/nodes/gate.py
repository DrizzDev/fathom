from __future__ import annotations

import time
from logging import getLogger
from typing import Any, Dict, Optional, cast

from fathom.constants.runtime import (
    DEFAULT_HEALING_RUN_BUDGET,
    DEFAULT_HEALING_TASK_BUDGET,
    DEFAULT_LOCALIZATION_BUDGET,
    DEFAULT_LOCALIZATION_CONFIDENCE_THRESHOLD,
    DEFAULT_PAID_LOCALIZATION_ATTEMPT_BUDGET,
)
from fathom.constants.screen import ZERO_HASH
from fathom.constants.state import CommonStateKey, IntentStateKey
from fathom.core.runtime.adapter import ExecutionTaskAdapter
from fathom.schemas.budgets import HealingBudget, LocalizationBudget
from fathom.schemas.effect import ActionEffect, ActionEffectStatus
from fathom.schemas.events import RuntimeEventKind
from fathom.schemas.healing import HealingDecisionKind, HealingRequest
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.supervision import SupervisionVerdict, VerdictKind
from fathom.schemas.tasks import ExecutionTask, ExecutionTaskState, TaskAttemptState
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes.persistence import GraphStatePersistence
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class ActionGate:
    """
    Runs localization, supervision, and bounded healing against a planned step.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        persistence: GraphStatePersistence,
    ) -> None:
        """
        Initialize the gate with the shared graph context and persistence helper.
        """

        self.__context = context
        self.__persistence = persistence

    async def localize(
        self,
        *,
        step: Step,
        capture: ScreenCapture,
        observation: ScreenObservation,
    ) -> LocalizationResult:
        """
        Resolve an action target against the runtime screen observation.
        """

        return await self.__context.target_localizer.localize(
            capture=capture,
            action=step.action,
            image=capture.image,
            observation=observation,
            budget=LocalizationBudget(
                vision=True,
                local=DEFAULT_LOCALIZATION_BUDGET,
                attempts=DEFAULT_PAID_LOCALIZATION_ATTEMPT_BUDGET,
                threshold=DEFAULT_LOCALIZATION_CONFIDENCE_THRESHOLD,
            ),
        )

    @staticmethod
    def apply_localization(*, step: Step, localization: LocalizationResult) -> Step:
        """
        Attach resolved localization evidence to an executable step.
        """

        if localization.status != LocalizationStatus.RESOLVED or localization.bounds is None:
            return step

        action = step.action.model_copy(update={"bounds": localization.bounds})
        return step.model_copy(update={"action": action})

    def supervise(
        self,
        *,
        step: Step,
        observation: ScreenObservation,
        localization: LocalizationResult,
    ) -> SupervisionVerdict:
        """
        Evaluate whether an action is allowed to execute.
        """

        return self.__context.runtime_supervisor.supervise(
            action=step.action,
            observation=observation,
            localization=localization,
            runtime=self.__context.agent_state.runtime,
        )

    def active_execution_task(self) -> ExecutionTask:
        """
        Return the active runtime task for healing context.
        """

        sub_goal = self.__context.agent_state.get_current_sub_goal()
        if sub_goal is not None:
            return ExecutionTaskAdapter().from_sub_goal(sub_goal=sub_goal)

        return ExecutionTask(
            identifier="task:implicit",
            state=ExecutionTaskState.ACTIVE,
            objective="Implicit runtime task",
            criterion="No explicit task criterion is available.",
            attempts=TaskAttemptState(count=0, limit=DEFAULT_HEALING_TASK_BUDGET),
        )

    async def heal_blocked_action(
        self,
        *,
        step: Step,
        capture: ScreenCapture,
        verdict: SupervisionVerdict,
        observation: ScreenObservation,
    ) -> Optional[Step]:
        """
        Return one bounded healing action for a blocked execution.
        """

        if verdict.reason is None:
            return None

        runtime = self.__context.agent_state.runtime
        execution_task = self.active_execution_task()
        task_id = execution_task.identifier

        decision = await self.__context.healing_orchestrator.decide(
            run_used=runtime.healing.run_count(),
            task_used=runtime.healing.task_count(task_id=task_id),
            failures=runtime.failures,
            request=HealingRequest(
                failed=(),
                capabilities=(),
                screen=observation,
                task=execution_task,
                reason=verdict.reason,
            ),
            budget=HealingBudget(
                run=DEFAULT_HEALING_RUN_BUDGET,
                task=DEFAULT_HEALING_TASK_BUDGET,
            ),
        )
        runtime.healing.record(task_id=task_id)

        await self.__context.event_emitter.emit(
            kind=RuntimeEventKind.HEALING_DECIDED,
            step=runtime.tasks.progress()[0],
            payload={
                "task.id": task_id,
                "decision.reason": decision.reason,
                "decision.kind": decision.kind.value,
                "run.healing.used": runtime.healing.run_count(),
                "block.reason": verdict.reason.value if verdict.reason else None,
                "task.healing.used": runtime.healing.task_count(task_id=task_id),
            },
        )
        logger.info(
            "Healing decision computed",
            extra={
                "task.id": task_id,
                "reason": decision.reason,
                "component": "graph.intent.gate",
                "event": "healing.decision.computed",
                "decision.kind": decision.kind.value,
                "workflow.id": self.__context.workflow_id,
                "run.healing.used": runtime.healing.run_count(),
                "task.healing.used": runtime.healing.task_count(task_id=task_id),
            },
        )

        if decision.kind != HealingDecisionKind.TRY_ACTION or decision.action is None:
            return None

        healed_action = await self.__context.resolution.substitute(action=decision.action)
        healed_step = step.model_copy(update={"action": healed_action})
        localization = await self.localize(
            capture=capture,
            step=healed_step,
            observation=observation,
        )
        healed_step = self.apply_localization(step=healed_step, localization=localization)
        healed_verdict = self.supervise(
            step=healed_step,
            observation=observation,
            localization=localization,
        )
        if healed_verdict.kind == VerdictKind.ALLOW:
            return healed_step

        logger.warning(
            "Healing action was blocked",
            extra={
                "component": "graph.intent.gate",
                "event": "healing.action.blocked",
                "reason": healed_verdict.reason.value if healed_verdict.reason else None,
                # `message` is reserved by LogRecord; use `verdict.message`
                # to surface the supervisor's human-readable detail.
                "verdict.message": healed_verdict.message,
            },
        )
        return None

    @staticmethod
    def blocked_step_result(
        *,
        step: Step,
        reason: str,
        start_time: float,
        capture: ScreenCapture,
    ) -> StepResult:
        """
        Build a failed step result for a pre-execution runtime block.
        """

        pre_hash = capture.state.visual_hash if capture.state is not None else ZERO_HASH

        return StepResult(
            step=step,
            error=reason,
            success=False,
            observation=None,
            pre_hash=pre_hash,
            post_hash=pre_hash,
            screen_changed=False,
            duration=int((time.time() - start_time) * 1000),
            generalized_target=step.action.script_target,
            is_positional=(step.action.target_type == "positional"),
        )

    def blocked_execute_result(
        self,
        *,
        step: Step,
        reason: str,
        start_time: float,
        capture: ScreenCapture,
        state: IntentGraphState,
    ) -> IntentGraphState:
        """
        Build graph state for a blocked action and record the no-progress effect.
        """

        step_result = self.blocked_step_result(
            step=step,
            reason=reason,
            capture=capture,
            start_time=start_time,
        )
        self.__context.agent_state.record_action_effect(
            effect=ActionEffect(
                scroll_dx=0.0,
                scroll_dy=0.0,
                ssim_score=1.0,
                phash_distance=0,
                content_change=0.0,
                visual_progress=0.0,
                status=ActionEffectStatus.NO_PROGRESS,
            )
        )

        blocked: Dict[Any, Any] = {
            CommonStateKey.STEP_RESULT: step_result,
            CommonStateKey.EXECUTION_DURATION: time.time() - start_time,
            IntentStateKey.ELEMENTS: state.get(IntentStateKey.ELEMENTS),
            CommonStateKey.SCREEN_OBSERVATION: state.get(CommonStateKey.SCREEN_OBSERVATION),
        }
        self.__persistence.persist(result=blocked)
        return cast("IntentGraphState", blocked)
