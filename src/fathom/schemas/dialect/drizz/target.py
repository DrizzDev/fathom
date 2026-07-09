from __future__ import annotations

from typing import Optional

from pydantic import Field

from fathom.constants.flow import CheckKind
from fathom.schemas.base import SealedModel


class Target(SealedModel):
    """
    A resolved UI target reference within a Drizz command.
    """

    text: str = Field(min_length=1, description="Exact target text.")
    position: Optional[str] = Field(default=None, description="Ordinal qualifier such as 'first'.")
    container: Optional[str] = Field(default=None, description="Container or section context.")


class Assertion(SealedModel):
    """
    A single validation assertion within a Validate command.
    """

    subject: str = Field(min_length=1, description="Subject being validated.")
    state: CheckKind = Field(default=CheckKind.VISIBLE, description="Asserted UI state.")
