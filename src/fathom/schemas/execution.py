from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.localization import LocalizationResult
from fathom.schemas.results import ExecutionResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step


class ExecutionContext(BaseModel):
    """
    Per-step execution context shared between SUPERVISE, EXECUTE, and OBSERVE.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    step: Step = Field(description="Action approved for execution by the supervisor.")
    capture: ScreenCapture = Field(description="Pre-action screen capture.")

    pre_screen: Optional[ScreenState] = Field(
        default=None,
        description="Pre-action screen state when available.",
    )
    localization: LocalizationResult = Field(
        description="Localization evidence carried from supervision.",
    )
    package: str = Field(description="Pre-action foreground package identifier.")
    execution_result: Optional[ExecutionResult] = Field(
        default=None,
        description="Result emitted by EXECUTE; populated by the execute node.",
    )
    duration: int = Field(
        default=0,
        description="Total execution duration in milliseconds; populated by EXECUTE.",
    )
