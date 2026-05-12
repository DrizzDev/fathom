from __future__ import annotations

import time
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FeedbackEntry(BaseModel):
    """
    Base for typed feedback entries flowing into ContextManager channels.
    Subclassed by source-specific schemas so each channel is type-distinct.
    """

    content: str = Field(description="Feedback text shown to downstream consumers")
    step_number: Optional[int] = Field(
        default=None, description="Agent step the entry was produced at"
    )
    timestamp: float = Field(
        default_factory=time.time, description="Wall-clock time of entry creation"
    )

    model_config = ConfigDict(frozen=True)


class UserGuidance(FeedbackEntry):
    """
    Instruction from a real human (HITL pause-inject or ASK_USER response).
    Rendered to the LLM with MUST-comply framing; run-scoped.
    """


class VerifierFeedback(FeedbackEntry):
    """
    Rejection reason emitted by the VERIFY node when the LLM-verifier rules
    a completion claim invalid. Use-once: consumed by the next planner
    iteration so the LLM can re-plan against the rejection.
    """
