from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.embedding import (
    DEFAULT_EMBEDDING_ATTEMPTS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_RETRY_BACKOFF,
    DEFAULT_EMBEDDING_TIMEOUT,
    EmbeddingProvider,
)


class EmbeddingRetryPolicy(BaseModel):
    """
    Retry policy applied when an embedding call times out or fails transiently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempts: int = Field(
        ge=1,
        default_factory=lambda: DEFAULT_EMBEDDING_ATTEMPTS,
        description="Total attempts including the first call.",
    )
    backoff: float = Field(
        ge=1.0,
        default_factory=lambda: DEFAULT_EMBEDDING_RETRY_BACKOFF,
        description="Multiplier applied between successive retry waits.",
    )


class EmbeddingConfiguration(BaseModel):
    """
    Boot-time configuration for an :class:`EmbeddingPort` implementation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        min_length=1,
        default_factory=lambda: DEFAULT_EMBEDDING_MODEL,
        description="Provider-specific model identifier.",
    )
    timeout: int = Field(
        ge=1,
        default_factory=lambda: DEFAULT_EMBEDDING_TIMEOUT,
        description="Per-attempt wait window in milliseconds.",
    )
    retry: EmbeddingRetryPolicy = Field(
        default_factory=EmbeddingRetryPolicy,
        description="Retry policy applied when an attempt times out.",
    )


class EmbeddingVector(BaseModel):
    """
    Single embedding vector produced for one input text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: Tuple[float, ...] = Field(
        min_length=1, description="Dense embedding values in row-major order."
    )

    def cosine(self, *, other: "EmbeddingVector") -> float:
        """
        Return cosine similarity between this vector and ``other`` on the unit interval.
        """

        if len(self.values) != len(other.values):
            raise ValueError(f"Vector length mismatch: {len(self.values)} vs {len(other.values)}")

        dot = sum(left * right for left, right in zip(self.values, other.values, strict=True))
        norm_left = sum(left * left for left in self.values) ** 0.5
        norm_right = sum(right * right for right in other.values) ** 0.5
        if norm_left == 0.0 or norm_right == 0.0:
            return 0.0
        return float(dot / (norm_left * norm_right))


class EmbeddingResult(BaseModel):
    """
    Result of one embedding request covering one or more input texts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: EmbeddingProvider = Field(description="Backend that produced the vectors.")
    model: str = Field(min_length=1, description="Provider model identifier.")
    duration: int = Field(ge=0, description="Provider call duration in milliseconds.")
    vectors: Tuple[EmbeddingVector, ...] = Field(
        min_length=1, description="One vector per input text, in request order."
    )
