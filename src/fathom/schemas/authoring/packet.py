from __future__ import annotations

from pydantic import Field

from fathom.schemas.authoring.reference import AuthoringDialectReference
from fathom.schemas.authoring.task import AuthoringTask
from fathom.schemas.base import SealedModel


class AuthoringPacket(SealedModel):
    """
    Complete typed input supplied to an authoring prompt strategy.
    """

    task: AuthoringTask = Field(description="Authoring task and evidence view.")
    dialect: AuthoringDialectReference = Field(description="Target dialect reference.")
