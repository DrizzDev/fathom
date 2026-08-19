from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fathom.constants.collaboration import ArtifactBackend
from fathom.constants.signing import SigningBackend, SigningStatus


class SigningRequest(BaseModel):
    """
    One artifact URI to sign, scoped to a storage backend.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(description="Stored artifact location.")
    backend: ArtifactBackend = Field(description="Storage backend that produced the URI.")


class SigningOutcome(BaseModel):
    """
    Typed result of one signing attempt.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(
        description=("Presigned URL when status is signed; the stored URI otherwise."),
    )
    status: SigningStatus = Field(description="Typed signing outcome.")


class GcsSigningConfiguration(BaseModel):
    """
    Credentials and URL lifetime the GCS signer uses to mint presigned artifact URLs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    credentials_json: Optional[str] = Field(
        default=None,
        description=(
            "Service-account credentials as a JSON string. When omitted, "
            "the signer falls back to Application Default Credentials."
        ),
    )
    ttl_seconds: int = Field(
        description="Presigned URL TTL in seconds.",
        ge=1,
    )


class S3SigningConfiguration(BaseModel):
    """
    Region and URL lifetime the S3 signer uses to mint presigned artifact URLs.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    region: str = Field(description="AWS region for the bucket.")
    ttl_seconds: int = Field(ge=1, description="Presigned URL TTL in seconds.")


class SigningConfiguration(BaseModel):
    """
    Deployment selection of signer backend plus its concrete configuration.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: SigningBackend = Field(description="Active signer family.")
    gcs: Optional[GcsSigningConfiguration] = Field(
        default=None,
        description="Required when backend is gcs; ignored otherwise.",
    )
    s3: Optional[S3SigningConfiguration] = Field(
        default=None,
        description="Required when backend is s3; ignored otherwise.",
    )

    @model_validator(mode="after")
    def __validate_backend_pairing(self) -> "SigningConfiguration":
        """
        Validate that the selected backend has its required configuration.
        """

        if self.backend == SigningBackend.GCS and self.gcs is None:
            raise ValueError("Missing GCS signer configuration.")

        if self.backend == SigningBackend.S3 and self.s3 is None:
            raise ValueError("Missing S3 signer configuration.")

        return self
