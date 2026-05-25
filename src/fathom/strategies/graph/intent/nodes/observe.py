from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from fathom.constants import ActionExecutionKind, ActionType
from fathom.constants.screen import ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD, ZERO_HASH
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenDiff
from fathom.schemas.steps import StepResult
from fathom.strategies.graph.intent.nodes.effect import PostAction
from fathom.strategies.graph.state import IntentGraphState

if TYPE_CHECKING:
    from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider

logger = logging.getLogger(__name__)


class ObserveNode:
    """
    OBSERVE graph node; captures post-action evidence and stages RECORD inputs.
    """

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider

    async def __call__(self, state: IntentGraphState) -> IntentGraphState:
        """
        Run the OBSERVE node handler.
        """

        return await self.run(state=state)

    async def run(self, *, state: IntentGraphState) -> IntentGraphState:
        """
        Capture the post-action screen and stage RECORD inputs.
        """

        logger.info(
            "Starting observe node",
            extra={
                "component": "graph.intent.observe",
                "event": "observe.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        if await self.__provider.is_cancelled():
            return self.__cancelled_result()

        context = state.get(IntentStateKey.EXECUTION_CONTEXT)
        if not isinstance(context, ExecutionContext) or context.execution_result is None:
            message = (
                "Observation failed: missing ExecutionContext or execution_result; "
                "EXECUTE did not commit a device result."
            )
            logger.error(
                message,
                extra={
                    "component": "graph.intent.observe",
                    "event": "observe.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
            result = cast(
                "IntentGraphState",
                {
                    IntentStateKey.SHOULD_RETRY: False,
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                    CommonStateKey.FAILURE_DIAGNOSTIC: message,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result
        execution_result = context.execution_result

        observation = state.get(CommonStateKey.SCREEN_OBSERVATION)
        if not isinstance(observation, ScreenObservation):
            observation = await self.__provider.observer.fallback_observation(
                state=state, capture=context.capture
            )

        (
            post_observation,
            screen_diff,
            post_hash,
            post_activity,
            step_artifacts,
        ) = await self.__provider.effects.observe(
            context=context,
        )

        if await self.__provider.is_cancelled():
            return self.__cancelled_result()

        pre_hash = context.pre_screen.visual_hash if context.pre_screen is not None else ZERO_HASH
        self.__provider.context.metrics.record(
            operation="action", duration=context.duration / 1000.0
        )

        action_effect = self.__provider.effects.effect_from(diff=screen_diff)
        if context.step.action.execution_kind is ActionExecutionKind.DEVICE:
            self.__provider.context.agent_state.record_action_effect(effect=action_effect)
        self.__provider.effects.log_diff(screen_diff=screen_diff, action_effect=action_effect)

        screen_changed = self.__screen_changed(
            context=context,
            post_hash=post_hash,
            pre_hash=pre_hash,
            screen_diff=screen_diff,
        )
        step_success = self.__step_success(
            action_execution_kind=context.step.action.execution_kind,
            execution_success=execution_result.success,
        )
        logger.info(
            "pre_hash=%s post_hash=%s screen_changed=%s",
            pre_hash[:8],
            post_hash[:8],
            screen_changed,
        )

        plan_observation = PostAction.plan_observation(state=state)
        step_result = StepResult(
            step=context.step,
            pre_hash=pre_hash,
            post_hash=post_hash,
            observation=plan_observation,
            duration=context.duration,
            error=execution_result.error,
            artifacts=step_artifacts,
            screen_changed=screen_changed,
            success=step_success,
            generalized_target=context.step.action.script_target,
            is_positional=(context.step.action.target_type == "positional"),
        )

        # The post-action observation is the authoritative snapshot for the
        # next ANALYZE turn. Persist it when we managed to capture one;
        # fall back to the pre-action observation only when the post capture
        # failed (so the graph never holds a None observation).
        next_observation = post_observation if post_observation is not None else observation

        result_dict: Dict[Any, Any] = {
            CommonStateKey.STEP_RESULT: step_result,
            CommonStateKey.EXECUTION_DURATION: context.duration / 1000.0,
            CommonStateKey.SCREEN_OBSERVATION: next_observation,
            IntentStateKey.ELEMENTS: state.get(IntentStateKey.ELEMENTS),
            IntentStateKey.POST_ACTIVITY: post_activity,
        }

        if context.step.action.action_type == ActionType.ASK_USER:
            logger.info(
                "Clearing graph state after ASK_USER for fresh analysis",
                extra={
                    "component": "graph.intent.observe",
                    "event": "observe.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            result_dict[IntentStateKey.PLAN] = None
            result_dict[CommonStateKey.IS_COMPLETE] = False
            result_dict[IntentStateKey.PLANNED_STEP] = None
            result_dict[IntentStateKey.SHOULD_RETRY] = True
            result_dict[CommonStateKey.COMPLETION_REASON] = None

        self.__provider.persistence.persist(result=result_dict)
        return result_dict  # type: ignore[return-value]

    def __cancelled_result(self) -> IntentGraphState:
        """
        Mark the workflow cancelled without recording another action result.
        """

        self.__provider.context.agent_state.mark_complete(reason=CompletionReason.CANCELLED.value)
        result = cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
            },
        )
        self.__provider.persistence.persist(result=result)
        return result

    def __screen_changed(
        self,
        *,
        context: ExecutionContext,
        pre_hash: str,
        post_hash: str,
        screen_diff: Optional[ScreenDiff],
    ) -> bool:
        """
        Return whether the canonical outcome recorded a screen change.
        """

        if context.step.action.execution_kind is not ActionExecutionKind.DEVICE:
            return False

        return self.__provider.effects.changed(
            screen_diff=screen_diff,
            pre_hash=pre_hash,
            post_hash=post_hash,
            threshold=ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD,
        )

    @staticmethod
    def __step_success(
        *,
        action_execution_kind: ActionExecutionKind,
        execution_success: bool,
    ) -> bool:
        """
        Return the canonical success bit for one recorded step.
        """

        _ = action_execution_kind
        return execution_success
