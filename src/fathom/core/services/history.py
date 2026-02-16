"""History service for persisting execution traces."""

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional

from fathom.schemas.steps import StepResult

logger = getLogger(__name__)


class HistoryService:
    """Service for saving execution history and steps."""

    def __init__(self, workflow_id: str = "default") -> None:
        self.__workflow_id = workflow_id

    def save_step(
        self, 
        result: StepResult, 
        absolute_center: Optional[List[int]] = None, 
        intent: str = ""
    ) -> None:
        """Save a step result to history."""
        logger.debug(f"Saving step {result.step.step_number} for workflow {self.__workflow_id}")
