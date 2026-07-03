from __future__ import annotations

import asyncio
import json
from logging import getLogger
from typing import Dict, Final
from urllib.parse import urlparse

from fathom.constants.collaboration import ArtifactBackend
from fathom.constants.signing import SigningStatus
from fathom.core.exceptions import ConfigurationError
from fathom.interfaces.signing import SigningPort
from fathom.schemas.signing import (
    GcsSigningConfiguration,
    SigningOutcome,
    SigningRequest,
)

try:
    from google.cloud import storage
    from google.oauth2 import service_account
except ModuleNotFoundError:
    storage = None
    service_account = None


class GcsSigner(SigningPort):
    """
    GCS artifact signer.

    Converts `gs://bucket/key` URIs into v4 presigned download URLs. All
    other backends and schemes pass through unchanged with the appropriate
    typed status.
    """

    __SCHEME: Final[str] = "gs"
    __SCHEME_HTTP: Final[str] = "http"
    __SCHEME_HTTPS: Final[str] = "https"
    __SIGNED_URL_METHOD: Final[str] = "GET"
    __SIGNED_URL_VERSION: Final[str] = "v4"

    def __init__(self, *, configuration: GcsSigningConfiguration) -> None:
        """
        Bind the GCS signer to one immutable configuration.
        """

        self.__configuration = configuration
        self.__logger = getLogger(".".join((__name__, self.__class__.__name__)))

    @property
    def ttl_seconds(self) -> int:
        """
        Return the configured signed-URL TTL in seconds.
        """

        return self.__configuration.ttl_seconds

    async def sign(self, *, request: SigningRequest) -> SigningOutcome:
        """
        Sign a GCS URI; preserve the stored URI on failure or non-GCS input.
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
                    event="artifact_signing_unsupported_scheme",
                    scheme=parsed.scheme,
                ),
            )
            return SigningOutcome(uri=request.uri, status=SigningStatus.UNSUPPORTED)

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            return SigningOutcome(uri=request.uri, status=SigningStatus.FAILED)

        try:
            signed = await asyncio.to_thread(
                self.__sign_synchronously,
                bucket,
                key,
            )
        except (RuntimeError, ValueError, OSError) as exception:
            self.__logger.exception(
                "GCS signing failed; preserving stored URI",
                extra=self.__log_extra(
                    event="artifact_signing_failed",
                    scheme=parsed.scheme,
                    error=type(exception).__name__,
                ),
            )
            return SigningOutcome(uri=request.uri, status=SigningStatus.FAILED)

        if signed == request.uri:
            self.__logger.warning(
                "GCS signing returned the stored URI unchanged",
                extra=self.__log_extra(
                    event="artifact_signing_no_change",
                    scheme=parsed.scheme,
                ),
            )
            return SigningOutcome(uri=request.uri, status=SigningStatus.FAILED)
        return SigningOutcome(uri=signed, status=SigningStatus.SIGNED)

    def __sign_synchronously(self, bucket_name: str, key: str) -> str:
        """
        Build a v4 signed URL synchronously; called off the event loop.
        """

        if storage is None or service_account is None:
            raise ConfigurationError(
                "GCS artifact signing requires google-cloud-storage and "
                "google-auth to be installed."
            )

        credentials_json = self.__configuration.credentials_json
        if credentials_json:
            credentials_info: Dict[str, object] = json.loads(credentials_json)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
            )
            client = storage.Client(credentials=credentials)
        else:
            client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        signed_url = blob.generate_signed_url(
            version=self.__SIGNED_URL_VERSION,
            expiration=self.__configuration.ttl_seconds,
            method=self.__SIGNED_URL_METHOD,
        )
        return str(signed_url)

    def __log_extra(self, *, event: str, **values: object) -> Dict[str, object]:
        """
        Build structured log context for signing decisions.
        """

        return {
            "component": "fathom_gcs_signer",
            "event": event,
            **values,
        }
