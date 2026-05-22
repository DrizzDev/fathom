from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional, cast

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.constants.command import CommandExecutionMode
from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.state import (
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
    PlanMetadataKey,
)
from fathom.core.recovery import (
    BoundedFailureOutcome,
    EscalateOutcome,
    RecoveryCoordinator,
    RecoveryOutcome,
    RecoveryRequest,
    RecoveryTrigger,
    ReplanOutcome,
    TryActionOutcome,
)
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.actions import Action
from fathom.schemas.escape import EscapeReport
from fathom.schemas.results import PlanResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.schemas.subgoal import RequiredActionFamily, ScrollAxis, SubGoal
from fathom.strategies.graph.context import GraphContext
from fathom.strategies.graph.intent.nodes.persistence import GraphStatePersistence
from fathom.strategies.graph.state import IntentGraphState

logger = getLogger(__name__)


class StrictExecutionContractGuard:
    """
    Validates strict-mode replans against structured execution contracts.
    """

    @classmethod
    def violation(
        cls,
        *,
        command_mode: CommandExecutionMode,
        current_sub_goal: Optional[SubGoal],
        replacement_sub_goals: List[SubGoal],
    ) -> Optional[str]:
        """
        Return a strict-mode violation when a replan changes the active execution contract.
        """

        if command_mode is not CommandExecutionMode.STRICT or current_sub_goal is None:
            return None

        contract = current_sub_goal.execution_contract
        if contract.required_action_family is RequiredActionFamily.UNSPECIFIED:
            return None

        allowed_families = cls.__allowed_families(
            required_action_family=contract.required_action_family
        )
        for replacement in replacement_sub_goals:
            replacement_family = replacement.execution_contract.required_action_family
            if replacement_family not in allowed_families:
                return (
                    "strict_replan_mismatch: active sub-goal requires "
                    f"{contract.required_action_family.value}-family recovery but replan proposed "
                    f"{replacement_family.value}-family step '{replacement.description}'"
                )

            if contract.surface and not cls.__same_surface(
                expected=contract.surface,
                observed=replacement.execution_contract.surface,
            ):
                return (
                    "strict_replan_mismatch: active sub-goal requires surface "
                    f"'{contract.surface}' but replan proposed surface "
                    f"'{replacement.execution_contract.surface or '(none)'}' in "
                    f"step '{replacement.description}'"
                )

            if contract.required_action_family is not RequiredActionFamily.SCROLL:
                continue

            if replacement_family is not RequiredActionFamily.SCROLL:
                continue

            replacement_axis = replacement.execution_contract.scroll_axis
            if contract.scroll_axis in {ScrollAxis.UNSPECIFIED, replacement_axis}:
                continue

            if replacement_axis is ScrollAxis.UNSPECIFIED:
                return (
                    "strict_replan_mismatch: active sub-goal requires "
                    f"{contract.scroll_axis.value} scroll recovery but replan omitted that axis in "
                    f"step '{replacement.description}'"
                )

            return (
                "strict_replan_mismatch: active sub-goal requires "
                f"{contract.scroll_axis.value} scroll recovery but replan proposed "
                f"{replacement_axis.value} scroll step '{replacement.description}'"
            )

        return None

    @staticmethod
    def __allowed_families(
        *, required_action_family: RequiredActionFamily
    ) -> set[RequiredActionFamily]:
        """
        Return the strict-mode family set allowed for one active contract.
        """

        if required_action_family is RequiredActionFamily.SCROLL:
            return {RequiredActionFamily.SCROLL}
        if required_action_family is RequiredActionFamily.TAP:
            return {RequiredActionFamily.TAP}
        if required_action_family is RequiredActionFamily.INPUT:
            return {RequiredActionFamily.INPUT, RequiredActionFamily.TAP}
        if required_action_family is RequiredActionFamily.WAIT:
            return {RequiredActionFamily.WAIT}
        if required_action_family is RequiredActionFamily.VALIDATE:
            return {RequiredActionFamily.VALIDATE}
        return {required_action_family}

    @staticmethod
    def __same_surface(*, expected: str, observed: Optional[str]) -> bool:
        """
        Compare structured surfaces using the repo's canonical normalization.
        """

        if not observed:
            return False

        return Normalizer.clean(text=expected).lower() == Normalizer.clean(text=observed).lower()


class EscapeReportDecoder:
    """
    Stateless decoder for planner-emitted metadata payloads.

    The decoder is intentionally free of dependencies — it consumes only
    a metadata dict and returns either a typed :class:`EscapeReport` or
    ``None``. Owns no context and no policy; pure transformation logic.
    """

    @staticmethod
    def extract(*, metadata: Optional[Dict[str, Any]]) -> Optional[EscapeReport]:
        """
        Decode an :class:`EscapeReport` from planner metadata.

        Returns ``None`` for missing metadata, missing key, non-dict
        payload, or schema-invalid payload. Any malformed input must
        degrade silently so the planner can keep producing turns while
        a deeper bug is investigated.
        """

        if not metadata or not isinstance(
            payload := metadata.get(PlanMetadataKey.ESCAPE_REPORT.value), dict
        ):
            return None
        try:
            return EscapeReport.model_validate(payload)
        except ValidationError as error:
            logger.warning(
                "Escape report metadata invalid",
                extra={
                    "component": "graph.intent.recovery",
                    "event": "escape.report.invalid",
                    "error.message": str(error),
                },
            )
            return None


class TraceDescriptorRenderer:
    """
    Stateless renderer for the recent-action trace handed to recovery strategies.

    The renderer is a pure transformation: ``(trace, window) → List[str]``.
    The dispatcher reads its policy window and trace from the live graph
    context but delegates the actual slicing and formatting here so the
    semantics can be pinned without forging a graph context in tests.
    """

    @staticmethod
    def render(*, trace: Any, window: int) -> List[str]:
        """
        Render the last ``window`` trace entries as one-line descriptors.

        The window is floored at one so a misconfigured ``recent_window=0``
        does not silently turn the whole trace into the recent slice
        (Python's ``list[-0:]`` returns every element). Non-dict trace
        entries are stringified rather than dropped so synthetic recovery
        steps stay observable.
        """

        if not isinstance(trace, list):
            return []

        floor = max(1, window)
        descriptors: List[str] = []
        for entry in list(trace)[-floor:]:
            if isinstance(entry, dict):
                action_kind = entry.get("action") or entry.get("action_type") or "action"
                target = entry.get("target") or entry.get("description") or ""
                descriptors.append(f"{action_kind}: {target}".strip())
            else:
                descriptors.append(str(entry))
        return descriptors


class RecoveryRequestBuilder:
    """
    Builds typed :class:`RecoveryRequest` instances from the live graph context.

    Reads sub-goal state, recent-action trace, and current screen hash
    from the injected :class:`GraphContext` and assembles the request
    every recovery strategy consumes. Owns no persistence and no
    outcome-translation logic — those live on the applier.
    """

    def __init__(self, *, context: GraphContext, coordinator: RecoveryCoordinator) -> None:
        """
        Bind the builder to the graph context and the coordinator. The
        coordinator is read solely for its ``policy.recent_window``.
        """

        self.__context = context
        self.__coordinator = coordinator

    def build(
        self,
        *,
        reason: str,
        capture: ScreenCapture,
        trigger: RecoveryTrigger,
        hint: Optional[str],
        escape_report: Optional[EscapeReport],
    ) -> Optional[RecoveryRequest]:
        """
        Build a :class:`RecoveryRequest` for the current sub-goal.

        Returns ``None`` when there is no active sub-goal — recovery
        only makes sense when the agent has a stuck objective.
        """

        agent_state = self.__context.agent_state
        if (current := agent_state.get_current_sub_goal()) is None:
            return None

        return RecoveryRequest(
            hint=hint,
            reason=reason,
            trigger=trigger,
            capture=capture,
            escape_report=escape_report,
            stuck_sub_goal=current.description,
            strict_mode=(
                self.__context.configuration.intent.command_mode is CommandExecutionMode.STRICT
            ),
            execution_contract=current.execution_contract,
            recent_actions=self.recent_action_descriptors(),
            # Mechanical strategies (overlay/keyboard/scroll) need the
            # current ScreenObservation to decide. Without this the
            # coordinator chain dies on the first observation-gated
            # strategy because every NoopOutcome reads "no observation
            # available" and recovery escalation never reaches replan.
            observation=agent_state.runtime.screen.observation,
            pending_sub_goals=[
                sub_goal.description
                for sub_goal in agent_state.sub_goal_list
                if not sub_goal.is_complete()
            ],
        )

    def recent_action_descriptors(self) -> List[str]:
        """
        Return the recent action descriptors capped to the policy window.

        Pulls the trace from the context manager, then delegates to the
        stateless :class:`TraceDescriptorRenderer`. Catches the narrow
        set of attribute/key/type errors that the context manager can
        raise without context — wider exceptions propagate so unexpected
        failures surface as failed runs, not silent empty traces.
        """

        try:
            full_context = self.__context.context_manager.get_full_context()
        except (AttributeError, KeyError, TypeError) as exception:
            logger.warning(
                "Recovery dispatcher could not read context trace",
                extra={
                    **self.__log_context(),
                    "event": "recovery.trace.unavailable",
                    "error.message": str(exception),
                },
            )
            return []

        return TraceDescriptorRenderer.render(
            trace=full_context.get("trace", []),
            window=self.__coordinator.policy.recent_window,
        )

    def current_screen_hash(self) -> str:
        """
        Return the truncated visual hash of the current screen.

        The hash seeds the synthetic step's :attr:`Step.screen_hash`
        when the outcome applier emits a recovery action.
        """

        current = self.__context.agent_state.current_screen
        if current is None or not current.visual_hash:
            return ""
        return current.visual_hash[:VISUAL_HASH_LENGTH]

    def __log_context(self) -> Dict[str, Any]:
        """
        Shared structured-logging context for builder entries.
        """

        return {
            "component": "graph.intent.recovery.builder",
            "workflow.id": self.__context.workflow_id,
        }


class RecoveryOutcomeApplier:
    """
    Translates committed :class:`RecoveryOutcome` instances into graph patches.

    Each :class:`RecoveryOutcome` subclass maps to a distinct patch
    shape: replan resets the pending sub-goal tail, try-action stages a
    synthetic plan, escalation emits an ASK_USER step, and bounded
    failure terminates the run. The applier owns the persistence helper
    so each patch is checkpointed before the node returns.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        builder: RecoveryRequestBuilder,
        persistence: GraphStatePersistence,
    ) -> None:
        """
        Bind the applier to the graph context, the request builder
        (for the current screen hash), and the persistence helper.
        """

        self.__context = context
        self.__builder = builder
        self.__persistence = persistence

    def apply(self, *, outcome: RecoveryOutcome) -> Optional[IntentGraphState]:
        """
        Dispatch the outcome to its typed apply method.

        Unknown outcome types return ``None`` so the dispatcher can
        treat them as no-op rather than crashing.
        """

        if isinstance(outcome, ReplanOutcome):
            return self.__apply_replan(outcome=outcome)
        if isinstance(outcome, TryActionOutcome):
            return self.__apply_try_action(outcome=outcome)
        if isinstance(outcome, EscalateOutcome):
            return self.__apply_escalate(outcome=outcome)
        if isinstance(outcome, BoundedFailureOutcome):
            return self.__apply_bounded_failure(outcome=outcome)
        return None

    def __apply_replan(self, *, outcome: ReplanOutcome) -> IntentGraphState:
        """
        Replace the pending sub-goal tail and route back to GROUND.
        """

        if (
            violation := StrictExecutionContractGuard.violation(
                command_mode=self.__context.configuration.intent.command_mode,
                current_sub_goal=self.__context.agent_state.get_current_sub_goal(),
                replacement_sub_goals=outcome.new_sub_goals,
            )
        ) is not None:
            logger.warning(
                "Recovery replan rejected by strict-mode mission guard",
                extra={
                    **self.__log_context(),
                    "event": "recovery.replan.rejected",
                    "summary": outcome.summary,
                    "reason": violation,
                },
            )
            result = cast(
                "IntentGraphState",
                {
                    CommonStateKey.IS_COMPLETE: False,
                    IntentStateKey.SHOULD_RETRY: True,
                    IntentStateKey.LAST_BLOCK_REASON: "strict_replan_mismatch",
                    IntentStateKey.LAST_BLOCK_MESSAGE: violation,
                },
            )
            self.__persistence.persist(result=result)
            return result

        self.__context.agent_state.replan_pending_sub_goals(new_sub_goals=outcome.new_sub_goals)
        self.__context.agent_state.reset_completion()
        self.__context.context_manager.clear_user_guidance()
        self.__context.context_manager.clear_verifier_feedback()

        logger.info(
            "Recovery replan dispatched",
            extra={
                **self.__log_context(),
                "event": "recovery.replan.applied",
                "summary": outcome.summary,
            },
        )
        result = cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.SHOULD_RETRY: True,
            },
        )
        self.__persistence.persist(result=result)
        return result

    def __apply_try_action(self, *, outcome: TryActionOutcome) -> IntentGraphState:
        """
        Stage the alternative action so EXECUTE retries with it next turn.
        """

        logger.info(
            "Recovery try-action dispatched",
            extra={
                **self.__log_context(),
                "event": "recovery.try_action.applied",
                "summary": outcome.summary,
            },
        )
        plan = PlanResult(
            step=Step(
                is_conditional=True,
                event_type="action",
                condition="recovery",
                action=outcome.action,
                screen_hash=self.__builder.current_screen_hash(),
                step_number=self.__context.agent_state.step_count,
            ),
            is_complete=False,
            should_retry=True,
            reason=outcome.summary,
        )
        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.PLAN: plan,
                IntentStateKey.SHOULD_RETRY: True,
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.PLANNED_STEP: plan.step,
            },
        )
        self.__persistence.persist(result=result)
        return result

    def __apply_escalate(self, *, outcome: EscalateOutcome) -> IntentGraphState:
        """
        Emit an ASK_USER step so EXECUTE routes the question to HITL.
        """

        logger.info(
            "Recovery escalation dispatched",
            extra={
                **self.__log_context(),
                "event": "recovery.escalation.applied",
                "summary": outcome.summary,
            },
        )
        action = Action(
            confidence=1.0,
            text=outcome.question,
            rationale=outcome.summary,
            target="Request user assistance",
            action_type=ActionType.ASK_USER,
        )
        plan = PlanResult(
            step=Step(
                action=action,
                is_conditional=True,
                event_type="action",
                condition="recovery",
                screen_hash=self.__builder.current_screen_hash(),
                step_number=self.__context.agent_state.step_count,
            ),
            is_complete=False,
            should_retry=True,
            reason=CompletionReason.INTERVENTION_REQUIRED.value,
        )
        result = cast(
            "IntentGraphState",
            {
                IntentStateKey.PLAN: plan,
                IntentStateKey.SHOULD_RETRY: True,
                CommonStateKey.IS_COMPLETE: False,
                IntentStateKey.PLANNED_STEP: plan.step,
            },
        )
        self.__persistence.persist(result=result)
        return result

    def __apply_bounded_failure(self, *, outcome: BoundedFailureOutcome) -> IntentGraphState:
        """
        Terminate the run with the supplied structured diagnostic.
        """

        logger.warning(
            "Recovery bounded-failure dispatched",
            extra={
                **self.__log_context(),
                "event": "recovery.bounded_failure.applied",
                "summary": outcome.summary,
                "diagnostic": outcome.diagnostic,
            },
        )
        self.__context.agent_state.mark_complete(reason=CompletionReason.FAILED.value)
        result = cast(
            "IntentGraphState",
            {
                CommonStateKey.IS_COMPLETE: True,
                CommonStateKey.COMPLETION_REASON: CompletionReason.FAILED.value,
                CommonStateKey.FAILURE_DIAGNOSTIC: outcome.diagnostic,
            },
        )
        self.__persistence.persist(result=result)
        return result

    def __log_context(self) -> Dict[str, Any]:
        """
        Shared structured-logging context for applier entries.
        """

        return {
            "component": "graph.intent.recovery.applier",
            "workflow.id": self.__context.workflow_id,
        }


class RecoveryDispatcher:
    """
    Thin orchestrator over the recovery sub-components.

    Composes :class:`RecoveryRequestBuilder`, :class:`RecoveryOutcomeApplier`,
    and :class:`EscapeReportDecoder` into the single entry-point node
    helpers use. Owns no business logic itself — every concern is
    delegated to one of the three single-responsibility sub-components.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
        coordinator: RecoveryCoordinator,
        persistence: GraphStatePersistence,
    ) -> None:
        """
        Compose the sub-components from the shared graph dependencies.
        """

        self.__context = context
        self.__coordinator = coordinator
        self.__builder = RecoveryRequestBuilder(context=context, coordinator=coordinator)
        self.__applier = RecoveryOutcomeApplier(
            context=context,
            builder=self.__builder,
            persistence=persistence,
        )

    async def try_recover(
        self,
        *,
        reason: str,
        capture: ScreenCapture,
        trigger: RecoveryTrigger,
        hint: Optional[str] = None,
        escape_report: Optional[EscapeReport] = None,
    ) -> Optional[IntentGraphState]:
        """
        Single entry point for graph-node recovery dispatch.

        Builds the request, sends it through the coordinator at the
        active sub-goal's scope, and applies the returned outcome.
        Returns ``None`` when there is no active sub-goal, when the
        coordinator declines to act, or when the outcome is unknown.
        """

        request = self.__builder.build(
            reason=reason,
            capture=capture,
            trigger=trigger,
            hint=hint,
            escape_report=escape_report,
        )
        if request is None:
            return None

        outcome = await self.__coordinator.handle(
            trigger=trigger,
            request=request,
            scope=self.__active_scope(),
        )
        if outcome is None:
            return None

        return self.__applier.apply(outcome=outcome)

    def apply(self, *, outcome: RecoveryOutcome) -> Optional[IntentGraphState]:
        """
        Apply an already-committed outcome.

        Exposed for callers that drive the coordinator out of band and
        only need the patch translation.
        """

        return self.__applier.apply(outcome=outcome)

    def recent_action_descriptors(self) -> List[str]:
        """
        Return the recent action descriptors capped to the policy window.

        Exposed for callers that build their own :class:`RecoveryRequest`
        outside the standard ``try_recover`` flow.
        """

        return self.__builder.recent_action_descriptors()

    def current_screen_hash(self) -> str:
        """
        Truncated visual hash of the current screen.
        """

        return self.__builder.current_screen_hash()

    @staticmethod
    def extract_escape_report(
        *,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[EscapeReport]:
        """
        Decode an :class:`EscapeReport` from planner metadata.

        Thin compatibility wrapper over :meth:`EscapeReportDecoder.extract`;
        kept here so existing call sites do not have to change import.
        """

        return EscapeReportDecoder.extract(metadata=metadata)

    @staticmethod
    def descriptors_from_trace(*, trace: Any, window: int) -> List[str]:
        """
        Render the last ``window`` trace entries as one-line descriptors.

        Thin compatibility wrapper over :meth:`TraceDescriptorRenderer.render`;
        kept here so existing call sites do not have to change import.
        """

        return TraceDescriptorRenderer.render(trace=trace, window=window)

    def __active_scope(self) -> int:
        """
        Return the active sub-goal index used as the coordinator scope.

        The builder validated that a sub-goal exists before we got here;
        defensively returns zero if the state mutated concurrently.
        """

        current = self.__context.agent_state.get_current_sub_goal()
        return current.index if current is not None else 0
