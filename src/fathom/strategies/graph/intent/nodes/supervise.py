from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, cast

from fathom.constants import ActionType
from fathom.constants.command import CommandExecutionMode
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import ElementSource
from fathom.schemas.resolution import ResolveStatus
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.scroll import ScrollLock
from fathom.schemas.steps import Step
from fathom.schemas.subgoal import RequiredActionFamily, ScrollAxis
from fathom.schemas.supervision import VerdictKind
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class SuperviseNode:
    """
    SUPERVISE graph node; localizes and gates the planned step.
    """

    def __init__(self, *, provider: IntentNodeProvider) -> None:
        """
        Bind the node to the shared intent provider.
        """

        self.__provider = provider

    async def __call__(self, state: IntentGraphState) -> IntentGraphState:
        """
        Run the SUPERVISE node handler.
        """

        return await self.run(state=state)

    async def run(self, *, state: IntentGraphState) -> IntentGraphState:
        """
        Substitute, localize, supervise, and (when blocked) heal the planned action.
        """

        logger.info(
            "Starting supervise node",
            extra={
                "component": "graph.intent.supervise",
                "event": "supervise.log",
                "workflow.id": self.__provider.context.workflow_id,
            },
        )

        if await self.__provider.is_cancelled():
            logger.warning(
                "Execution cancelled",
                extra={
                    "component": "graph.intent.supervise",
                    "event": "supervise.log",
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            self.__provider.context.agent_state.mark_complete(
                reason=CompletionReason.CANCELLED.value
            )
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: True,
                    CommonStateKey.COMPLETION_REASON: CompletionReason.CANCELLED.value,
                },
            )
            self.__provider.persistence.persist(result=result)
            return result

        screen_capture = state.get(CommonStateKey.CAPTURE)
        planned_step = state.get(IntentStateKey.PLANNED_STEP)
        if not isinstance(planned_step, Step) or not isinstance(screen_capture, ScreenCapture):
            # Upstream (GROUND or ANALYZE) failed to publish the capture or the planned step.
            # Without both, SUPERVISE cannot localize a target — and neither can EXECUTE / OBSERVE.
            # Signal a re-ground via SHOULD_RETRY so the router takes us back to GROUND (bounded by max_steps),
            logger.warning(
                "Supervise: upstream state incomplete; routing back to GROUND",
                extra={
                    "component": "graph.intent.supervise",
                    "event": "supervise.upstream.invalid",
                    "has_planned_step": isinstance(planned_step, Step),
                    "workflow.id": self.__provider.context.workflow_id,
                    "has_capture": isinstance(screen_capture, ScreenCapture),
                },
            )
            retry_patch = cast(
                "IntentGraphState",
                {IntentStateKey.SHOULD_RETRY: True},
            )
            self.__provider.persistence.persist(result=retry_patch)
            return retry_patch

        observation = await self.__provider.observer.fallback_observation(
            state=state, capture=screen_capture
        )
        planned_step = self.__apply_scroll_lock(step=planned_step, state=state)
        elements = self.__elements_from_state(state=state)
        if (strict_violation := self.__command_mode_violation(step=planned_step)) is not None:
            return self.__strict_mode_blocked_patch(
                capture=screen_capture,
                reason=strict_violation,
                state=state,
                step=planned_step,
            )

        if planned_step.action.action_type == ActionType.ASK_USER:
            return self.__allow_non_spatial_step(
                state=state,
                step=planned_step,
                capture=screen_capture,
                localization=LocalizationResult(
                    status=LocalizationStatus.UNRESOLVED,
                    bounds=None,
                    source=None,
                    confidence=0.0,
                    reason="ask_user_bypass",
                ),
            )

        resolve_result = await self.__provider.context.resolution.resolve(
            action=planned_step.action,
            elements=elements,
        )
        step = planned_step.model_copy(update={"action": resolve_result.action})

        if resolve_result.status == ResolveStatus.RESOLVED:
            # Perception cascade — Stage 1: snap_to_label against the
            # merged manifest. ``state[ELEMENTS]`` contains XML, OCR,
            # icon, and CV entries thanks to ManifestMerger, so a
            # successful snap may be sourced from any perception layer,
            # not only XML. We synthesize a RESOLVED LocalizationResult
            # so the runtime supervisor sees the same shape it would
            # after a vision-localizer call.
            logger.info(
                "Perception cascade Stage 1 (snap) committed",
                extra={
                    "component": "graph.intent.supervise",
                    "event": "supervise.cascade.stage1.resolved",
                    "label_id": step.action.label_id,
                    "target": step.action.target,
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            localization = self.__localization_from_snap(step=step)
        else:
            # Stage 2: vision-localizer (Gemini) + local-localizer
            # (DocumentAI / icon templates / overlay pixels) via the
            # gate. Runs when the manifest snap could not bind the
            # target to a concrete bounds.
            logger.info(
                "Perception cascade Stage 2 (vision-localizer) engaged",
                extra={
                    "component": "graph.intent.supervise",
                    "event": "supervise.cascade.stage2.engaged",
                    "reason": resolve_result.reason,
                    "target": step.action.target,
                    "label_id": step.action.label_id,
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            localization = await self.__provider.gate.localize(
                step=step,
                capture=screen_capture,
                observation=observation,
            )
            step = self.__provider.gate.apply_localization(step=step, localization=localization)

        verdict = self.__provider.gate.supervise(
            step=step,
            localization=localization,
            observation=observation,
        )

        if verdict.kind != VerdictKind.ALLOW:
            logger.warning(
                "runtime supervision blocked action: %s",
                verdict.message,
                extra={
                    "component": "supervise",
                    "event": "runtime_blocked",
                    "reason": verdict.reason.value if verdict.reason else None,
                    "target": step.action.target,
                    "label_id": step.action.label_id,
                },
            )

            healed_step = await self.__provider.gate.heal_blocked_action(
                step=step,
                capture=screen_capture,
                observation=observation,
                verdict=verdict,
            )
            if healed_step is None:
                blocked = self.__provider.gate.blocked_execute_result(
                    step=step,
                    capture=screen_capture,
                    start_time=time.time(),
                    reason=(
                        "target_unresolved"
                        if localization.status
                        in {LocalizationStatus.UNRESOLVED, LocalizationStatus.AMBIGUOUS}
                        else "runtime_blocked"
                    )
                    + f": {verdict.message}",
                    state=state,
                )
                blocked_patch = cast(
                    "IntentGraphState",
                    {
                        **blocked,
                        IntentStateKey.EXECUTION_BLOCKED: True,
                        # Feedback for the next planner turn: the LLM
                        # sees the BlockReason and the supervisor's
                        # message rendered as <LAST_ACTION_BLOCK>.
                        IntentStateKey.LAST_BLOCK_REASON: (
                            verdict.reason.value if verdict.reason else None
                        ),
                        IntentStateKey.LAST_BLOCK_MESSAGE: verdict.message,
                    },
                )
                self.__provider.persistence.persist(result=blocked_patch)
                return blocked_patch

            logger.info(
                "approving healed action: %s",
                healed_step.action.to_description(),
                extra={
                    "component": "supervise",
                    "event": "healed_action",
                    "action_type": healed_step.action.action_type.value,
                    "target": healed_step.action.target,
                },
            )
            step = healed_step

        return self.__allow_non_spatial_step(
            state=state,
            step=step,
            capture=screen_capture,
            localization=localization,
        )

    def __command_mode_violation(self, *, step: Step) -> Optional[str]:
        """
        Return a rejection reason when the planned step drifts outside the active sub-goal command family.
        """

        if (
            self.__provider.context.configuration.intent.command_mode
            is not CommandExecutionMode.STRICT
        ):
            return None

        current = self.__provider.context.agent_state.get_current_sub_goal()
        if current is None:
            return None

        contract = current.execution_contract
        required_action_family = contract.required_action_family
        if required_action_family is RequiredActionFamily.UNSPECIFIED:
            return None

        action_type = step.action.action_type

        if self.__is_terminal_validation_candidate(step=step):
            return None

        allowed = self.__allowed_action_types(
            required_action_family=required_action_family,
            scroll_axis=contract.scroll_axis,
        )
        if action_type not in allowed:
            return (
                "strict_command_mismatch: active sub-goal requires "
                f"{required_action_family.value}-family actions but planner proposed "
                f"'{action_type.value}'"
            )

        if contract.surface and not self.__surface_matches_contract(
            expected=contract.surface,
            observed=step.action.surface,
            action_type=action_type,
        ):
            return (
                "strict_command_mismatch: active sub-goal requires surface "
                f"'{contract.surface}' but planner proposed "
                f"surface '{step.action.surface or '(none)'}'"
            )

        return None

    @staticmethod
    def __allowed_action_types(
        *,
        required_action_family: RequiredActionFamily,
        scroll_axis: ScrollAxis,
    ) -> set[ActionType]:
        """
        Return the action-type set allowed by one structured strict-mode contract.
        """

        if required_action_family is RequiredActionFamily.SCROLL:
            scroll_actions = {ActionType.SCROLL}
            if scroll_axis in {ScrollAxis.UNSPECIFIED, ScrollAxis.VERTICAL}:
                scroll_actions.update({ActionType.SWIPE_UP, ActionType.SWIPE_DOWN})
            if scroll_axis in {ScrollAxis.UNSPECIFIED, ScrollAxis.HORIZONTAL}:
                scroll_actions.update({ActionType.SWIPE_LEFT, ActionType.SWIPE_RIGHT})
            return scroll_actions | {ActionType.ASK_USER}

        if required_action_family is RequiredActionFamily.TAP:
            return {
                ActionType.TAP,
                ActionType.LONG_PRESS,
                ActionType.ASK_USER,
            }

        if required_action_family is RequiredActionFamily.INPUT:
            return {
                ActionType.TAP,
                ActionType.TYPE,
                ActionType.ASK_USER,
            }

        if required_action_family is RequiredActionFamily.WAIT:
            return {
                ActionType.WAIT,
                ActionType.ASK_USER,
            }

        if required_action_family is RequiredActionFamily.VALIDATE:
            return {
                ActionType.VALIDATE,
                ActionType.ASK_USER,
            }

        return set(ActionType)

    @staticmethod
    def __surface_matches_contract(
        *,
        expected: str,
        observed: Optional[str],
        action_type: ActionType,
    ) -> bool:
        """
        Enforce one explicit surface contract when the planner surfaced one on the action.
        """

        if action_type is ActionType.ASK_USER:
            return True
        if not observed:
            return False
        return Normalizer.clean(text=expected).lower() == Normalizer.clean(text=observed).lower()

    @staticmethod
    def __is_terminal_validation_candidate(*, step: Step) -> bool:
        """
        Allow one validate action through strict mode when the planner is using it
        only as a terminal completion claim for the active mission.
        """

        if step.action.action_type is not ActionType.VALIDATE:
            return False

        return bool(step.metadata.get("terminal_validation_candidate"))

    def __strict_mode_blocked_patch(
        self,
        *,
        capture: ScreenCapture,
        reason: str,
        state: IntentGraphState,
        step: Step,
    ) -> IntentGraphState:
        """
        Build the blocked patch for one strict command-family violation.
        """

        blocked = self.__provider.gate.blocked_execute_result(
            step=step,
            capture=capture,
            start_time=time.time(),
            reason=reason,
            state=state,
        )
        blocked_patch = cast(
            "IntentGraphState",
            {
                **blocked,
                IntentStateKey.EXECUTION_BLOCKED: True,
                IntentStateKey.LAST_BLOCK_REASON: "strict_command_mismatch",
                IntentStateKey.LAST_BLOCK_MESSAGE: reason,
            },
        )
        self.__provider.persistence.persist(result=blocked_patch)
        return blocked_patch

    @staticmethod
    def __elements_from_state(*, state: IntentGraphState) -> Optional[Dict[str, Any]]:
        """
        Read the drawer label-map out of state for snap-to-label.

        Returns ``None`` when the manifest hasn't been produced yet so
        :meth:`ReferenceResolutionService.resolve` can route to an
        ``UNRESOLVED`` outcome instead of crashing on attribute access.
        """

        raw = state.get(IntentStateKey.ELEMENTS)
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def __localization_from_snap(*, step: Step) -> LocalizationResult:
        """
        Synthesize a :class:`LocalizationResult` matching the snapped
        action so the supervisor sees the same shape it would after a
        successful Gemini-vision localization.

        Surfaces ``source=XML`` (the manifest snap is the source of
        truth) with full confidence so the supervisor's confidence
        gates treat it as a first-class localization.
        """

        bounds = step.action.bounds
        return LocalizationResult(
            status=LocalizationStatus.RESOLVED,
            bounds=bounds,
            source=ElementSource.XML,
            confidence=1.0,
        )

    def __allow_non_spatial_step(
        self,
        *,
        state: IntentGraphState,
        step: Step,
        capture: ScreenCapture,
        localization: LocalizationResult,
    ) -> IntentGraphState:
        """
        Build execution context for a step that should bypass normal gating.
        """

        package_name = self.__provider.context.package_name or "unknown"
        current_screen_state = state.get(CommonStateKey.SCREEN_STATE)
        if package_name == "unknown" and isinstance(current_screen_state, ScreenState):
            package_name = current_screen_state.activity or "unknown"

        execution_context = ExecutionContext(
            step=step,
            capture=capture,
            pre_screen=(
                current_screen_state if isinstance(current_screen_state, ScreenState) else None
            ),
            localization=localization,
            package=package_name,
        )

        observation = state.get(CommonStateKey.SCREEN_OBSERVATION)
        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.EXECUTION_CONTEXT: execution_context,
                CommonStateKey.SCREEN_OBSERVATION: observation,
                IntentStateKey.PLANNED_STEP: step,
                IntentStateKey.LAST_BLOCK_REASON: None,
                IntentStateKey.LAST_BLOCK_MESSAGE: None,
                IntentStateKey.EXECUTION_BLOCKED: False,
            },
        )
        self.__provider.persistence.persist(result=result)
        return result

    def __apply_scroll_lock(self, *, step: Step, state: IntentGraphState) -> Step:
        """
        Reuse the previously resolved scroll container for the same active scroll objective.
        """

        action_type = step.action.action_type.value.lower()
        if not action_type.startswith("swipe_"):
            return step

        raw_lock = state.get(IntentStateKey.ACTIVE_SCROLL_LOCK)
        if not isinstance(raw_lock, ScrollLock):
            return step

        locked_target = Normalizer.clean(text=raw_lock.target).lower()
        current_target = Normalizer.clean(
            text=step.action.scroll_target
            or step.action.natural_language_target
            or step.action.target
        ).lower()
        if not self.__targets_match(locked_target=locked_target, current_target=current_target):
            return step

        if not self.__axis_matches_lock(action_type=action_type, lock=raw_lock):
            return step

        logger.info(
            "Reusing locked scroll container for repeated objective",
            extra={
                "component": "graph.intent.supervise",
                "event": "supervise.scroll.lock.applied",
                "target": current_target,
                "scope.identifier": raw_lock.scope.identifier,
            },
        )
        return step.model_copy(
            update={
                "action": step.action.model_copy(
                    update={
                        "bounds": raw_lock.scope.bounds,
                        "label_id": None,
                    }
                )
            }
        )

    @staticmethod
    def __axis_matches_lock(*, action_type: str, lock: ScrollLock) -> bool:
        """
        Return whether the proposed swipe action preserves the locked axis and direction family.
        """

        direction = lock.direction.value.lower()
        if direction in {"up", "down"}:
            return action_type.endswith("_up") or action_type.endswith("_down")

        return action_type.endswith("_left") or action_type.endswith("_right")

    @staticmethod
    def __targets_match(*, locked_target: str, current_target: str) -> bool:
        """
        Return whether two scroll target phrases are semantically close enough to reuse the lock.
        """

        if not locked_target or not current_target:
            return False

        if locked_target == current_target:
            return True

        if locked_target in current_target or current_target in locked_target:
            return True

        locked_tokens = set(locked_target.split())
        current_tokens = set(current_target.split())
        if not locked_tokens or not current_tokens:
            return False

        overlap = len(locked_tokens & current_tokens)
        return overlap >= max(1, min(len(locked_tokens), len(current_tokens)) // 2)
