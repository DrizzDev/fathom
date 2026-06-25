from __future__ import annotations

from enum import StrEnum
from typing import Optional, Type

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.llm import InferencePriorityTransitionReason, InferenceTier
from fathom.schemas.base.common import ThresholdConfiguration


class StructuredOutput(BaseModel):
    """
    Vendor-neutral specification for constrained-decoding LLM output.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    payload: Type[BaseModel] = Field(
        description="Pydantic model the emitted response must conform to.",
    )


class GeminiExceptionKind(StrEnum):
    """
    Classified Gemini failure categories used for retry and recovery decisions.
    """

    GENERIC = "generic"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    PROVIDER_OVERLOADED = "provider_overloaded"
    STALE_CACHED_CONTENT = "stale_cached_content"


class GeminiExceptionMetadata(BaseModel):
    """
    Normalized Gemini exception metadata used for retry and cache recovery decisions.
    """

    exception_type: str = Field(description="Python exception type name")
    message: str = Field(description="Rendered provider exception message")
    status_code: Optional[int] = Field(default=None, description="HTTP-style provider status")

    retry_after_seconds: Optional[float] = Field(
        default=None,
        description="Retry-After delay when returned by the provider",
    )
    kind: GeminiExceptionKind = Field(
        default=GeminiExceptionKind.GENERIC,
        description="Normalized Gemini exception classification",
    )


class PriorityInferenceSignal(BaseModel):
    """
    Provider-neutral outcome signal from one LLM attempt.
    """

    model_config = ConfigDict(frozen=True)

    tier: InferenceTier = Field(description="Inference tier used for the attempt.")
    success: bool = Field(description="Whether the provider returned a usable response.")
    transient: bool = Field(
        default=False,
        description="Whether a failure appears intermittent and worth elevated capacity.",
    )
    latency: Optional[float] = Field(
        default=None,
        description="Elapsed provider call time for successful responses.",
    )


class PriorityInferenceEvidence(BaseModel):
    """
    Provider-neutral evidence that explains an adaptive priority decision.
    """

    model_config = ConfigDict(frozen=True)

    window: int = Field(description="Retained signal count used for the decision.")
    failures: int = Field(description="Transient failures seen in the retained window.")
    slows: int = Field(description="Slow successful responses seen in the retained window.")
    healthy: int = Field(description="Consecutive healthy priority responses.")
    threshold: ThresholdConfiguration = Field(description="Thresholds used for the decision.")


class PriorityInferenceTransition(BaseModel):
    """
    Provider-neutral adaptive priority tier transition.
    """

    model_config = ConfigDict(frozen=True)

    previous: InferenceTier = Field(description="Tier selected before the transition.")
    current: InferenceTier = Field(description="Tier selected after the transition.")
    reason: InferencePriorityTransitionReason = Field(description="Reason for the transition.")
    evidence: PriorityInferenceEvidence = Field(description="Evidence that caused the transition.")
