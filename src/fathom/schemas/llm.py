from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


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
