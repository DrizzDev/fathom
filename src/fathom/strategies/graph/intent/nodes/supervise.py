from __future__ import annotations

import logging
from typing import Any, Dict, Optional, cast

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import ElementSource
from fathom.schemas.resolution import ResolveStatus
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.provider import IntentNodeProvider
from fathom.strategies.graph.state import IntentGraphState

logger = logging.getLogger(__name__)


class SuperviseNode:
    """
    SUPERVISE graph node; resolves/localizes the planned step.
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
        Substitute, localize, and build the execution context for the planned action.
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
            # not only XML.
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
            snap_outcome = resolve_result.unresolved_kind
            logger.info(
                "Perception cascade Stage 2 (vision-localizer) engaged",
                extra={
                    "component": "graph.intent.supervise",
                    "event": "supervise.cascade.stage2.engaged",
                    "target": step.action.target,
                    "reason": resolve_result.reason,
                    "label_id": step.action.label_id,
                    "snap.outcome": snap_outcome.value if snap_outcome is not None else None,
                    "workflow.id": self.__provider.context.workflow_id,
                },
            )
            localization = await self.__provider.gate.localize(
                step=step,
                capture=screen_capture,
                observation=observation,
                snap_outcome=snap_outcome,
            )
            step = self.__provider.gate.apply_localization(step=step, localization=localization)

            if self.__requires_localization_retry(step=step, localization=localization):
                return self.__retry_unresolved_spatial_step(
                    step=step,
                    localization=localization,
                    reason=resolve_result.reason,
                )

        return self.__allow_non_spatial_step(
            step=step,
            state=state,
            capture=screen_capture,
            localization=localization,
        )

    @staticmethod
    def __elements_from_state(*, state: IntentGraphState) -> Optional[Dict[str, Any]]:
        """
        Read the drawer label-map out of state for snap-to-label.

        Returns ``None`` when the manifest hasn't been produced yet so :meth:`ReferenceResolutionService.resolve`
        can route to an ``UNRESOLVED`` outcome instead of crashing on attribute access.
        """

        raw = state.get(IntentStateKey.ELEMENTS)
        return raw if isinstance(raw, dict) else None

    def __requires_localization_retry(
        self,
        *,
        step: Step,
        localization: LocalizationResult,
    ) -> bool:
        """
        Return whether unresolved spatial model coordinates must be re-planned.
        """

        action_type = step.action.action_type
        catalog = self.__provider.context.catalog

        return (
            catalog.is_spatial(action_type=action_type)
            and not catalog.is_gesture(action_type=action_type)
            and localization.status != LocalizationStatus.RESOLVED
        )

    def __retry_unresolved_spatial_step(
        self,
        *,
        step: Step,
        reason: Optional[str],
        localization: LocalizationResult,
    ) -> IntentGraphState:
        """
        Route unresolved element actions back to GROUND instead of executing guessed coordinates.
        """

        diagnostic = (
            "Spatial action could not be grounded; refusing to execute model-only coordinates. "
            f"target={step.action.target!r}; resolution={reason or 'unresolved'}; "
            f"localization={localization.reason or localization.status.value}"
        )

        logger.warning(
            "Supervise refused unresolved spatial action",
            extra={
                "component": "graph.intent.supervise",
                "event": "supervise.spatial.unresolved",
                "target": step.action.target,
                "action.type": step.action.action_type.value,
                "localization.status": localization.status.value,
                "workflow.id": self.__provider.context.workflow_id,
            },
        )
        self.__provider.context.agent_state.record_attempt(
            action=step.action,
            reason="supervise_spatial_unresolved",
        )
        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.SHOULD_RETRY: True,
                IntentStateKey.PLANNED_STEP: step,
                IntentStateKey.EXECUTION_CONTEXT: None,
                CommonStateKey.FAILURE_DIAGNOSTIC: diagnostic,
            },
        )
        self.__provider.persistence.persist(result=result)
        return result

    @staticmethod
    def __localization_from_snap(*, step: Step) -> LocalizationResult:
        """
        Synthesize a :class:`LocalizationResult` matching the snapped action.

        Surfaces ``source=XML`` (the manifest snap is the source of truth)
        with full confidence.
        """

        bounds = step.action.bounds
        return LocalizationResult(
            bounds=bounds,
            confidence=1.0,
            source=ElementSource.XML,
            status=LocalizationStatus.RESOLVED,
        )

    def __allow_non_spatial_step(
        self,
        *,
        step: Step,
        capture: ScreenCapture,
        state: IntentGraphState,
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
            package=package_name,
            localization=localization,
            pre_screen=(
                current_screen_state if isinstance(current_screen_state, ScreenState) else None
            ),
        )

        observation = state.get(CommonStateKey.SCREEN_OBSERVATION)
        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.PLANNED_STEP: step,
                CommonStateKey.SCREEN_OBSERVATION: observation,
                IntentStateKey.EXECUTION_CONTEXT: execution_context,
            },
        )
        self.__provider.persistence.persist(result=result)

        return result
