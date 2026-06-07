from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.abort import (
    DEFAULT_ABORT_CONFIDENCE_FLOOR,
    DEFAULT_ABORT_DETECTOR_MAX_RETRIES,
    DEFAULT_ABORT_DETECTOR_MODEL,
    DEFAULT_ABORT_DETECTOR_TEMPERATURE,
    DEFAULT_ABORT_DETECTOR_TIMEOUT,
    DEFAULT_ABORT_FALLBACK_SIMILARITY_FLOOR,
)


class AbortDecision(BaseModel):
    """
    Outcome of a single operator-abort classification.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    aborted: bool = Field(description="True when the response commands the agent to stop.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classifier confidence in the verdict.",
    )
    fallback: bool = Field(
        default=False,
        description="True when the classifier abstained and a safe default was returned.",
    )


class AbortDetectorResponse(BaseModel):
    """
    Raw LLM response parsed at the abort-detector adapter boundary.
    """

    model_config = ConfigDict(extra="forbid")

    aborted: bool = Field(description="Whether the operator response commands a workflow stop.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classifier-reported confidence.",
    )


class AbortInferenceConfiguration(BaseModel):
    """
    Inference parameters for the abort-detector LLM call.
    """

    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        default=DEFAULT_ABORT_DETECTOR_MODEL,
        description="LLM model identifier.",
    )
    temperature: float = Field(
        default=DEFAULT_ABORT_DETECTOR_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    timeout: int = Field(
        gt=0,
        default=DEFAULT_ABORT_DETECTOR_TIMEOUT,
        description="Per-attempt timeout in milliseconds.",
    )
    max_retries: int = Field(
        ge=0,
        default=DEFAULT_ABORT_DETECTOR_MAX_RETRIES,
        description="Maximum retry attempts on transient errors.",
    )
    use_cache: bool = Field(
        default=False,
        description="Whether to reuse cached LLM responses across calls.",
    )


class AbortConfidenceConfiguration(BaseModel):
    """
    Confidence policy for accepting an LLM-returned abort verdict.
    """

    model_config = ConfigDict(extra="forbid")

    floor: float = Field(
        ge=0.0,
        le=1.0,
        default=DEFAULT_ABORT_CONFIDENCE_FLOOR,
        description="Minimum confidence required to honour an aborted verdict.",
    )


class AbortFallbackConfiguration(BaseModel):
    """
    Heuristic fallback policy used when the LLM classifier abstains.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Whether to invoke the heuristic fallback when the LLM abstains.",
    )
    similarity_floor: float = Field(
        ge=0.0,
        le=1.0,
        default=DEFAULT_ABORT_FALLBACK_SIMILARITY_FLOOR,
        description="Minimum fuzzy similarity required to honour an abort match.",
    )


class AbortDetectorConfiguration(BaseModel):
    """
    Top-level configuration for the abort-detector pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Master toggle for the abort detector.",
    )
    confidence: AbortConfidenceConfiguration = Field(
        default_factory=AbortConfidenceConfiguration,
        description="Confidence policy for the LLM verdict.",
    )
    fallback: AbortFallbackConfiguration = Field(
        default_factory=AbortFallbackConfiguration,
        description="Heuristic-fallback policy when the LLM abstains.",
    )
    inference: AbortInferenceConfiguration = Field(
        default_factory=AbortInferenceConfiguration,
        description="LLM inference parameters.",
    )
