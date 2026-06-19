from __future__ import annotations

from typing import Optional

from fathom.constants.runtime import (
    DEFAULT_LOCALIZATION_BUDGET,
    DEFAULT_LOCALIZATION_CONFIDENCE_THRESHOLD,
    DEFAULT_PAID_LOCALIZATION_ATTEMPT_BUDGET,
)
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.resolution import UnresolvedKind
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step
from fathom.strategies.graph.context import GraphContext


class ActionGate:
    """
    Runs target localization against a planned step.
    """

    def __init__(
        self,
        *,
        context: GraphContext,
    ) -> None:
        """
        Initialize the gate with the shared graph context.
        """

        self.__context = context

    async def localize(
        self,
        *,
        step: Step,
        capture: ScreenCapture,
        observation: ScreenObservation,
        snap_outcome: Optional[UnresolvedKind] = None,
    ) -> LocalizationResult:
        """
        Resolve an action target against the runtime screen observation.
        """

        return await self.__context.target_localizer.localize(
            capture=capture,
            action=step.action,
            image=capture.image,
            observation=observation,
            snap_outcome=snap_outcome,
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
