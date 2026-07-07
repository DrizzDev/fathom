from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.constants.conversation import THREAD_TITLE_MAX_LENGTH
from fathom.schemas.base import SealedModel


class TitleContext(SealedModel):
    """
    Runtime context used to compose a conversation title.
    """

    intent: str = Field(description="User intent supplied for semantic title generation.")
    package: Optional[str] = Field(
        default=None,
        description="Application package used for deterministic fallback titles.",
    )


class TitlePolicy(SealedModel):
    """
    Bounds and fallback wording applied to generated conversation titles.
    """

    limit: int = Field(
        ge=1,
        default=THREAD_TITLE_MAX_LENGTH,
        description="Stored title character boundary.",
    )
    phrase: int = Field(
        ge=1,
        default=80,
        description="Generated action-phrase character boundary.",
    )
    token: int = Field(
        ge=1,
        default=36,
        description="Whitespace-free token boundary for generated titles.",
    )
    fallback: str = Field(
        min_length=1,
        default="Authoring session",
        description="Fallback title when no application package is available.",
    )
    prefix: str = Field(
        min_length=1,
        default="Authoring",
        description="Prefix used for package-derived fallback titles.",
    )
