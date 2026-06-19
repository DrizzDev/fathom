from __future__ import annotations

import time
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.reasoning import USER_GUIDANCE_ANALYZE_TTL


class GuidanceStatus(StrEnum):
    """
    Lifecycle state for human instructions injected during a run.
    """

    ACTIVE = "active"
    EXPIRED = "expired"


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
    Rendered to the LLM with MUST-comply framing while active.
    """

    status: GuidanceStatus = Field(
        default=GuidanceStatus.ACTIVE,
        description="Whether this instruction is still eligible for prompt rendering.",
    )
    remaining_analyses: int = Field(
        default=USER_GUIDANCE_ANALYZE_TTL,
        ge=0,
        description="Number of future ANALYZE turns where this instruction may appear.",
    )

    @property
    def active(self) -> bool:
        """
        Return whether the guidance should be rendered to the planner.
        """

        return self.status == GuidanceStatus.ACTIVE and self.remaining_analyses > 0

    def consume(self) -> "UserGuidance":
        """
        Return this guidance after one planner exposure.
        """

        remaining = max(0, self.remaining_analyses - 1)
        status = GuidanceStatus.ACTIVE if remaining > 0 else GuidanceStatus.EXPIRED
        return self.model_copy(update={"remaining_analyses": remaining, "status": status})

    def render(self) -> str:
        """
        Render bounded guidance for prompt inclusion.
        """

        return f"[active, remaining_analyze_turns={self.remaining_analyses}] {self.content}"


class VerifierFeedback(FeedbackEntry):
    """
    Rejection reason emitted by the VERIFY node when the LLM-verifier rules
    a completion claim invalid. Use-once: consumed by the next planner
    iteration so the LLM can adjust the next action against the rejection.
    """
