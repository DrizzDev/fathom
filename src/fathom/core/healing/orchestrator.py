from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional

from fathom.constants import ActionType
from fathom.constants.healing import (
    KEYBOARD_DISMISS_RATIONALE,
    KEYBOARD_DISMISS_TARGET,
    OVERLAY_DISMISS_RATIONALE,
    VISIBLE_ACTION_RATIONALE,
)
from fathom.core.runtime.failures import FailureMemory
from fathom.interfaces.healing import HealingAgentPort
from fathom.schemas.actions import Action
from fathom.schemas.budgets import HealingBudget
from fathom.schemas.configuration import IntentConfiguration
from fathom.schemas.healing import HealingDecision, HealingDecisionKind, HealingRequest
from fathom.schemas.observation import PerceivedElement
from fathom.schemas.perception import PerceptionConfiguration
from fathom.schemas.supervision import BlockReason

logger = getLogger(__name__)


class HealingOrchestrator:
    """
    Produces bounded healing decisions for blocked runtime execution.
    """

    def __init__(
        self,
        *,
        workflow_id: Optional[str] = None,
        agent: Optional[HealingAgentPort] = None,
        perception_configuration: Optional[PerceptionConfiguration] = None,
        runtime_policy: Optional[IntentConfiguration.RuntimePolicyConfiguration] = None,
    ) -> None:
        """
        Initialize the orchestrator with an optional agentic healer and run context.
        """

        self.__agent = agent
        self.__workflow_id = workflow_id
        self.__perception_configuration = perception_configuration or PerceptionConfiguration()
        self.__runtime_policy = runtime_policy or IntentConfiguration.RuntimePolicyConfiguration()

    async def decide(
        self,
        *,
        run_used: int,
        task_used: int,
        budget: HealingBudget,
        request: HealingRequest,
        failures: Optional[FailureMemory] = None,
    ) -> HealingDecision:
        """
        Decide how to recover from a blocked runtime state under healing budgets.
        """

        context = self.__log_context(request=request)

        if task_used >= budget.task or run_used >= budget.run:
            logger.info(
                "Healing budget exhausted",
                extra={
                    **context,
                    "run.used": run_used,
                    "task.used": task_used,
                    "run.budget": budget.run,
                    "task.budget": budget.task,
                    "event": "healing.budget.exhausted",
                },
            )
            return HealingDecision(
                kind=HealingDecisionKind.FAIL_BOUNDED,
                reason="Healing budget exhausted for this run or task.",
            )

        if (mechanical := self.__mechanical(request=request, failures=failures)) is not None:
            logger.info(
                "Mechanical healing decision",
                extra={
                    **context,
                    "event": "healing.mechanical.decided",
                    "decision.kind": mechanical.kind.value,
                },
            )
            return mechanical

        if self.__agent is not None:
            logger.info(
                "Delegating to agentic healer",
                extra={**context, "event": "healing.agent.invoked"},
            )
            return await self.__agent.decide(request=request)

        logger.info(
            "Healing fell through with no decision",
            extra={**context, "event": "healing.fell.through"},
        )
        return HealingDecision(
            kind=HealingDecisionKind.FAIL_BOUNDED,
            reason="No mechanical recovery available and no healing agent configured.",
        )

    def __mechanical(
        self,
        *,
        request: HealingRequest,
        failures: Optional[FailureMemory],
    ) -> Optional[HealingDecision]:
        """
        Return a deterministic healing decision when one is obvious and unblocked.
        """

        if (
            self.__perception_configuration.keyboard.enabled
            and self.__runtime_policy.keyboard.allow_recovery
            and request.reason == BlockReason.KEYBOARD_OCCLUDING
        ):
            action = Action(
                confidence=1.0,
                target=KEYBOARD_DISMISS_TARGET,
                action_type=ActionType.HIDE_KEYBOARD,
                rationale=KEYBOARD_DISMISS_RATIONALE,
                natural_language_target=KEYBOARD_DISMISS_TARGET,
            )

            if self.__blocked(action=action, failures=failures):
                return None

            return HealingDecision(
                action=action,
                reason=KEYBOARD_DISMISS_RATIONALE,
                kind=HealingDecisionKind.TRY_ACTION,
            )

        if (
            request.reason == BlockReason.OVERLAY_STILL_PRESENT
            and (
                candidate := self.__first_unblocked_overlay_candidate(
                    request=request, failures=failures
                )
            )
            is not None
        ):
            return self.__tap_candidate(
                candidate=candidate,
                reason=OVERLAY_DISMISS_RATIONALE,
            )

        if (
            request.reason == BlockReason.TARGET_UNRESOLVED
            and (candidate := self.__single_visible_action(request=request)) is not None
            and not self.__blocked(
                failures=failures,
                action=self.__candidate_action(candidate=candidate),
            )
        ):
            return self.__tap_candidate(
                candidate=candidate,
                reason=VISIBLE_ACTION_RATIONALE,
            )

        if request.reason in {
            BlockReason.REPEATED_NO_EFFECT,
            BlockReason.NON_SCROLLABLE_SURFACE,
        }:
            return None

        return None

    def __blocked(self, *, action: Action, failures: Optional[FailureMemory]) -> bool:
        """
        Return whether failure memory already marks the action as ineffective.
        """

        if failures is None:
            return False

        return failures.is_blocked(action=action)

    def __single_visible_action(self, *, request: HealingRequest) -> Optional[PerceivedElement]:
        """
        Return the only visible call-to-action when deterministic.
        """

        if len(request.screen.calls_to_action) != 1:
            return None

        return request.screen.calls_to_action[0]

    def __first_unblocked_overlay_candidate(
        self,
        *,
        request: HealingRequest,
        failures: Optional[FailureMemory],
    ) -> Optional[PerceivedElement]:
        """
        Return the first overlay dismiss candidate not already known to be ineffective.
        """

        for overlay in request.screen.overlays:
            for candidate in overlay.candidates:
                if self.__blocked(
                    failures=failures,
                    action=self.__candidate_action(candidate=candidate),
                ):
                    continue

                return candidate

        return None

    def __first_unblocked_visible_action(
        self,
        *,
        request: HealingRequest,
        failures: Optional[FailureMemory],
    ) -> Optional[PerceivedElement]:
        """
        Return the first visible call-to-action not already known to be ineffective.
        """

        for candidate in request.screen.calls_to_action:
            if self.__blocked(
                failures=failures,
                action=self.__candidate_action(candidate=candidate),
            ):
                continue

            return candidate

        return None

    def __tap_candidate(self, *, candidate: PerceivedElement, reason: str) -> HealingDecision:
        """
        Build a tap decision for a perceived element.
        """

        return HealingDecision(
            reason=reason,
            kind=HealingDecisionKind.TRY_ACTION,
            action=self.__candidate_action(candidate=candidate, reason=reason),
        )

    @staticmethod
    def __candidate_action(
        *,
        candidate: PerceivedElement,
        reason: str = VISIBLE_ACTION_RATIONALE,
    ) -> Action:
        """
        Build a tap Action from a perceived element for failure-memory comparison.
        """

        target = candidate.text or candidate.identifier

        return Action(
            target=target,
            rationale=reason,
            bounds=candidate.bounds,
            action_type=ActionType.TAP,
            label_id=candidate.identifier,
            natural_language_target=target,
            confidence=candidate.confidence,
        )

    def __log_context(self, *, request: HealingRequest) -> Dict[str, Any]:
        """
        Return shared structured-logging context for healing entries.
        """

        return {
            "workflow.id": self.__workflow_id,
            "activity": request.screen.activity,
            "component": "core.healing.orchestrator",
            "task.id": request.task.identifier if request.task is not None else None,
            "block.reason": request.reason.value if request.reason is not None else None,
        }
