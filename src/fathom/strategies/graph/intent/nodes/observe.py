from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from fathom.constants import ActionExecutionKind, ActionType
from fathom.constants.screen import ACTION_EFFECT_PHASH_DISTANCE_THRESHOLD, ZERO_HASH
from fathom.constants.scroll import ScrollDirection
from fathom.constants.state import CommonStateKey, IntentStateKey
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.outcomes import ActionOutcome, OutcomeStatus
from fathom.schemas.screens import ScreenDiff
from fathom.schemas.scroll import ScrollLock
from fathom.schemas.steps import StepResult
from fathom.strategies.graph.intent.nodes.effect import PostAction
from fathom.strategies.graph.state import IntentGraphState

if TYPE_CHECKING:
    from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider

logger = logging.getLogger(__name__)


class ObserveNode:
    """
    OBSERVE graph node; captures post-action evidence and classifies outcome.
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
        Capture the post-action screen, classify the outcome, and stage RECORD inputs.
        """

        logger.info(
            "Starting observe node",
            extra={
                "component": "graph.intent.observe",
                "event": "observe.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        if state.get(IntentStateKey.EXECUTION_BLOCKED):
            logger.info(
                "Skipping observe: supervisor blocked execution",
                extra={
                    "component": "graph.intent.observe",
                    "event": "observe.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            return cast("IntentGraphState", {})

        context = state.get(IntentStateKey.EXECUTION_CONTEXT)
        if not isinstance(context, ExecutionContext) or context.execution_result is None:
            logger.error(
                "Missing execution context or result; cannot observe",
                extra={
                    "component": "graph.intent.observe",
                    "event": "observe.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            return cast("IntentGraphState", {})
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

        pre_hash = context.pre_screen.visual_hash if context.pre_screen is not None else ZERO_HASH
        self.__provider.context.metrics.record(
            operation="action", duration=context.duration / 1000.0
        )

        action_outcome = self.__provider.context.outcome_classifier.classify(
            action=context.step.action,
            before=observation,
            after=post_observation,
            diff=screen_diff,
            success=execution_result.success,
            scroll_outcome=execution_result.scroll_outcome,
        )
        logger.info(
            "action outcome=%s reason=%s",
            action_outcome.status.value,
            action_outcome.reason,
            extra={
                "component": "observe",
                "event": "action_outcome",
                "outcome": action_outcome.status.value,
                "reason": action_outcome.reason,
            },
        )

        action_effect = self.__provider.effects.effect_from(
            diff=screen_diff,
            status=action_outcome.status,
        )
        if context.step.action.execution_kind is ActionExecutionKind.DEVICE:
            self.__provider.context.agent_state.record_action_effect(effect=action_effect)
        self.__provider.effects.log_diff(screen_diff=screen_diff, action_effect=action_effect)

        screen_changed = self.__screen_changed(
            action_outcome=action_outcome,
            context=context,
            post_hash=post_hash,
            pre_hash=pre_hash,
            screen_diff=screen_diff,
        )
        step_success = self.__step_success(
            action_outcome=action_outcome,
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
            CommonStateKey.ACTION_OUTCOME: action_outcome,
            CommonStateKey.SCREEN_OBSERVATION: next_observation,
            IntentStateKey.ELEMENTS: state.get(IntentStateKey.ELEMENTS),
            IntentStateKey.POST_ACTIVITY: post_activity,
            IntentStateKey.ACTIVE_SCROLL_LOCK: self.__scroll_lock_from_context(context=context),
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

    def __screen_changed(
        self,
        *,
        action_outcome: ActionOutcome,
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

        if action_outcome.status is OutcomeStatus.EFFECTIVE:
            return True

        if action_outcome.status is not OutcomeStatus.UNKNOWN:
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
        action_outcome: ActionOutcome,
        action_execution_kind: ActionExecutionKind,
        execution_success: bool,
    ) -> bool:
        """
        Return the canonical success bit for one recorded step.
        """

        if action_execution_kind is ActionExecutionKind.CONTROL:
            return execution_success

        return action_outcome.status is OutcomeStatus.EFFECTIVE

    @staticmethod
    def __scroll_lock_from_context(*, context: ExecutionContext) -> ScrollLock | None:
        """
        Persist one stable scroll lock for repeated swipe-like objectives.
        """

        execution_result = context.execution_result
        if execution_result is None or execution_result.scroll_outcome is None:
            return None

        if execution_result.scroll_outcome.scope is None:
            return None

        action_type = context.step.action.action_type.value.lower()
        if not action_type.startswith("swipe_"):
            return None

        target = (
            context.step.action.scroll_target
            or context.step.action.natural_language_target
            or context.step.action.target
        )
        if not target:
            return None

        if action_type.endswith("up"):
            direction = ScrollDirection.DOWN
        elif action_type.endswith("down"):
            direction = ScrollDirection.UP
        elif action_type.endswith("left"):
            direction = ScrollDirection.RIGHT
        else:
            direction = ScrollDirection.LEFT

        return ScrollLock(
            scope=execution_result.scroll_outcome.scope,
            direction=direction,
            target=target,
        )
