from __future__ import annotations

from logging import getLogger
from typing import Optional, Tuple

from fathom.core.capability.catalog import CommandCatalog
from fathom.core.services.binding import Binder
from fathom.schemas.binding import Binding
from fathom.schemas.localization import LocalizationResult
from fathom.schemas.observation import PerceivedElement
from fathom.schemas.steps import Step

logger = getLogger(__name__)


class GroundingRecorder:
    """
    Produces and records the typed grounding result for a supervised step without touching dispatch.
    """

    def __init__(self, *, binder: Optional[Binder] = None) -> None:
        """
        Bind the grounding producer whose results are recorded.
        """

        self.__binder = binder if binder is not None else Binder()

    def observe(
        self,
        *,
        step: Step,
        workflow_id: str,
        catalog: CommandCatalog,
        elements: Tuple[PerceivedElement, ...],
        localization: LocalizationResult,
    ) -> Optional[Binding]:
        """
        Ground the step's spatial target and log the result; never raise into the live loop.
        """

        if not catalog.is_spatial(action_type=step.action.action_type):
            return None

        try:
            binding = self.__binder.bind(
                action=step.action,
                elements=elements,
                localization=localization,
            )
        except Exception as exception:
            logger.warning(
                "Grounding production failed; dispatch unaffected",
                extra={
                    "event": "binding.failed",
                    "workflow.id": workflow_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return None

        logger.info(
            "Binding produced",
            extra={
                "event": "binding.produced",
                "workflow.id": workflow_id,
                "action.label_id": step.action.label_id,
                "action.type": step.action.action_type.value,
                "binding": binding.model_dump(mode="json"),
                "binding.moved": binding.bounds is not None
                and binding.bounds != step.action.bounds,
            },
        )
        return binding
