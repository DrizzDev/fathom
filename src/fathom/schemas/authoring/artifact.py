from __future__ import annotations

from typing import Optional, Tuple

from pydantic import Field

from fathom.constants.authoring import AuthoringArtifactKind, AuthoringArtifactRole
from fathom.constants.dialect import DialectName
from fathom.schemas.base import SealedModel
from fathom.schemas.flow import Issue
from fathom.schemas.generation import ScriptLineage


class AuthoringArtifact(SealedModel):
    """
    Authored output produced for a target dialect.
    """

    dialect: DialectName = Field(description="Dialect of the authored output.")
    kind: AuthoringArtifactKind = Field(description="Artifact category.")
    content: str = Field(min_length=1, description="Rendered artifact content.")
    advisories: Tuple[Issue, ...] = Field(
        default_factory=tuple, description="Non-blocking authoring review notes."
    )
    lineage: Tuple[ScriptLineage, ...] = Field(
        default_factory=tuple, description="Evidence provenance for authored script nodes."
    )


class AuthoringArtifactReference(SealedModel):
    """
    Reference to an artifact available to authoring without embedding bytes.
    """

    kind: AuthoringArtifactKind = Field(description="Artifact category.")
    role: AuthoringArtifactRole = Field(description="How the artifact should be used.")
    uri: str = Field(min_length=1, description="Dereferenceable artifact URI or local path.")

    mime: Optional[str] = Field(default=None, description="Artifact media type when known.")
    step_index: Optional[int] = Field(
        default=None, ge=0, description="Execution step this artifact belongs to."
    )
