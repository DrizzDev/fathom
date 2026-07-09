from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.schemas.base import SealedModel
from fathom.schemas.execution import ExecutionContext
from fathom.schemas.screens import ScreenCapture, ScreenDiff, ScreenHashBundle, ScreenState


class PreActionScreen(SealedModel):
    """
    Screen state captured before an action was dispatched.
    """

    capture: ScreenCapture = Field(description="Screen captured before action execution.")
    state: Optional[ScreenState] = Field(
        default=None,
        description="Comparable pre-action screen state when available.",
    )


class PostActionScreen(SealedModel):
    """
    Screen state captured immediately after an action was dispatched.
    """

    diff: ScreenDiff = Field(description="Initial before/after screen comparison.")
    capture: ScreenCapture = Field(description="Initial post-action screen capture.")
    hashes: ScreenHashBundle = Field(description="Hashes for the initial post-action capture.")


class ScreenSettlementEvidence(SealedModel):
    """
    Evidence required to run one bounded screen-settlement comparison.
    """

    execution: ExecutionContext = Field(description="Execution context for the observed action.")
    workflow_id: Optional[str] = Field(
        default=None,
        description="Workflow identifier for structured observation logs.",
    )

    before: PreActionScreen = Field(description="Pre-action screen evidence.")
    after: PostActionScreen = Field(description="Initial post-action screen evidence.")


class ScreenSettlement(SealedModel):
    """
    Screen evidence selected after the settlement pass.
    """

    capture: ScreenCapture = Field(description="Selected post-action capture.")
    diff: ScreenDiff = Field(description="Comparison for the selected post-action capture.")
    hashes: ScreenHashBundle = Field(description="Hashes for the selected post-action capture.")
