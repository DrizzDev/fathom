from __future__ import annotations

from typing import Optional, Tuple

from fathom.constants import StepEvent
from fathom.schemas.actions import Action
from fathom.schemas.requirement import CommandRequirement
from fathom.schemas.results import PlanContext, PlanResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.schemas.tools import StateUpdate, ToolArtifact, ToolData, ToolDiagnostic


class PlanStepFactory:
    """
    Constructs typed steps and plan results from an admitted action and its screen.
    """

    def plan_result(
        self,
        *,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        memories: int = 0,
        is_recovery: bool = False,
        context: Optional[PlanContext] = None,
        metrics: Optional[dict[str, float]] = None,
        updates: Tuple[StateUpdate, ...] = (),
        data: Tuple[ToolData, ...] = (),
        artifacts: Tuple[ToolArtifact, ...] = (),
        diagnostics: Tuple[ToolDiagnostic, ...] = (),
        requirement: Optional[CommandRequirement] = None,
    ) -> PlanResult:
        """
        Return a PlanResult carrying a constructed step and the turn's typed context.
        """

        step = self.step(
            action=action,
            capture=capture,
            requirement=requirement,
            is_recovery=is_recovery,
            step_number=step_number,
            event_type=action.event_type,
        )

        return PlanResult(
            step=step,
            is_complete=False,
            memories=memories,
            metrics=metrics or {},
            context=context if context is not None else PlanContext(),
            updates=updates,
            data=data,
            artifacts=artifacts,
            diagnostics=diagnostics,
            is_valid_action=action.is_valid,
            validation_reasoning=action.validation_reason,
            reason=action.rationale or ("Step planned" if not is_recovery else "Recovery step"),
        )

    def step(
        self,
        *,
        action: Action,
        step_number: int,
        capture: ScreenCapture,
        is_recovery: bool = False,
        event_type: Optional[StepEvent] = None,
        requirement: Optional[CommandRequirement] = None,
    ) -> Step:
        """
        Construct a Step, carrying only an admission-verified command requirement.
        """

        return Step(
            action=action,
            screen_hash=capture.identity,
            requirement=requirement,
            step_number=step_number,
            is_conditional=is_recovery,
            event_type=event_type,
            condition="recovery" if is_recovery else None,
        )
