from __future__ import annotations

from typing import Tuple

from pydantic import Field

from fathom.schemas.base import SealedModel
from fathom.schemas.dialect.drizz.command import DrizzCommand


class DrizzScript(SealedModel):
    """
    An ordered sequence of typed Drizz commands.
    """

    commands: Tuple[DrizzCommand, ...] = Field(
        default_factory=tuple, description="Ordered commands."
    )
