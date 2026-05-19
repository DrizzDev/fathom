from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, cast

from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import ElementSource
from fathom.schemas.resolution import ResolveStatus
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step
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
        elements = self.__elements_from_state(state=state)

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

        package_name = self.__provider.context.package_name or "unknown"
        current_screen_state = state.get(CommonStateKey.SCREEN_STATE)
        if package_name == "unknown" and isinstance(current_screen_state, ScreenState):
            package_name = current_screen_state.activity or "unknown"

        execution_context = ExecutionContext(
            step=step,
            capture=screen_capture,
            pre_screen=(
                current_screen_state if isinstance(current_screen_state, ScreenState) else None
            ),
            localization=localization,
            package=package_name,
        )

        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.EXECUTION_CONTEXT: execution_context,
                CommonStateKey.SCREEN_OBSERVATION: observation,
                IntentStateKey.PLANNED_STEP: step,
                # Successful allow path: clear any block memo so the
                # next planner turn does not see a stale block hint.
                IntentStateKey.LAST_BLOCK_REASON: None,
                IntentStateKey.LAST_BLOCK_MESSAGE: None,
            },
        )
        self.__provider.persistence.persist(result=result)
        return result

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
