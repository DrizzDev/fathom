from __future__ import annotations

import asyncio
from logging import getLogger
from typing import Dict, Final, Optional
from urllib.parse import urlparse

from fathom.constants.collaboration import ArtifactBackend
from fathom.constants.signing import SigningStatus
from fathom.core.exceptions import ConfigurationError
from fathom.interfaces.signing import SigningPort
from fathom.schemas.signing import (
    S3SigningConfiguration,
    SigningOutcome,
    SigningRequest,
)

try:
    import aioboto3
    from botocore.config import Config as _BotocoreConfig
except ModuleNotFoundError:
    aioboto3 = None
    _BotocoreConfig = None


class S3Signer(SigningPort):
    """
    S3 artifact signer.

    Converts `s3://bucket/key` URIs into v4 presigned download URLs. All
    other backends and schemes pass through unchanged with the appropriate typed status.
    """

    __SCHEME: Final[str] = "s3"
    __SCHEME_HTTP: Final[str] = "http"
    __SCHEME_HTTPS: Final[str] = "https"
    __SIGNATURE_VERSION: Final[str] = "s3v4"
    __CLIENT_METHOD: Final[str] = "get_object"
    __ADDRESSING_STYLE: Final[str] = "virtual"

    def __init__(self, *, configuration: S3SigningConfiguration) -> None:
        """
        Bind the S3 signer to one immutable configuration.
        """

        self.__configuration = configuration
        self.__logger = getLogger(".".join((__name__, self.__class__.__name__)))

        self.__session_lock = asyncio.Lock()
        self.__session: Optional["aioboto3.Session"] = None

    @property
    def ttl_seconds(self) -> int:
        """
        Return the configured signed-URL TTL in seconds.
        """

        return self.__configuration.ttl_seconds

    async def sign(self, *, request: SigningRequest) -> SigningOutcome:
        """
        Sign an S3 URI; preserve the stored URI on failure or non-S3 input.
        """

        if request.backend != ArtifactBackend.OBJECT:
            return SigningOutcome(uri=request.uri, status=SigningStatus.NOT_REQUIRED)

        parsed = urlparse(request.uri)

        if parsed.scheme in (self.__SCHEME_HTTP, self.__SCHEME_HTTPS):
            return SigningOutcome(uri=request.uri, status=SigningStatus.NOT_REQUIRED)

        if parsed.scheme != self.__SCHEME:
            self.__logger.warning(
                "Unsupported artifact object URI scheme",
                extra=self.__log_extra(
                    scheme=parsed.scheme,
                    event="artifact_signing_unsupported_scheme",
                ),
            )
            return SigningOutcome(uri=request.uri, status=SigningStatus.UNSUPPORTED)

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket or not key:
            return SigningOutcome(uri=request.uri, status=SigningStatus.FAILED)

        try:
            signed = await self.__presign(bucket=bucket, key=key)
        except (RuntimeError, ValueError, OSError) as exception:
            self.__logger.exception(
                "S3 signing failed; preserving stored URI",
                extra=self.__log_extra(
                    scheme=parsed.scheme,
                    error=type(exception).__name__,
                    event="artifact_signing_failed",
                ),
            )
            return SigningOutcome(uri=request.uri, status=SigningStatus.FAILED)

        if signed == request.uri:
            self.__logger.warning(
                "S3 signing returned the stored URI unchanged",
                extra=self.__log_extra(
                    scheme=parsed.scheme,
                    event="artifact_signing_no_change",
                ),
            )
            return SigningOutcome(uri=request.uri, status=SigningStatus.FAILED)

        return SigningOutcome(uri=signed, status=SigningStatus.SIGNED)

    async def __presign(self, *, bucket: str, key: str) -> str:
        """
        Build the presigned URL via the cached aioboto3 session.
        """

        if aioboto3 is None or _BotocoreConfig is None:
            raise ConfigurationError("S3 artifact signing requires aioboto3 to be installed.")

        if self.__session is None:
            async with self.__session_lock:
                if self.__session is None:
                    self.__session = aioboto3.Session()

        config = _BotocoreConfig(
            region_name=self.__configuration.region,
            signature_version=self.__SIGNATURE_VERSION,
            s3={"addressing_style": self.__ADDRESSING_STYLE},
        )
        async with self.__session.client(
            "s3",
            config=config,
            region_name=self.__configuration.region,
        ) as client:
            signed_url = await client.generate_presigned_url(
                ClientMethod=self.__CLIENT_METHOD,
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=self.__configuration.ttl_seconds,
            )
            return str(signed_url)

    def __log_extra(self, *, event: str, **values: object) -> Dict[str, object]:
        """
        Build structured log context for signing decisions.
        """

        return {
            **values,
            "event": event,
            "component": "fathom_s3_signer",
        }
