from __future__ import annotations

from fathom.adapters.signing.gcs import GcsSigner
from fathom.adapters.signing.noop import NoopSigner
from fathom.adapters.signing.s3 import S3Signer
from fathom.constants.signing import SigningBackend
from fathom.core.exceptions import ConfigurationError
from fathom.interfaces.signing import SigningPort
from fathom.schemas.signing import SigningConfiguration


class SignerFactory:
    """
    Build the signer adapter chosen by deployment configuration.

    Hosts call `SignerFactory.build` once at startup with a validated
    `SigningConfiguration` and hand the resulting port to `ConversationService`.
    """

    @classmethod
    def build(cls, *, configuration: SigningConfiguration) -> SigningPort:
        """
        Return the signer adapter matching `configuration.backend`.
        """

        if configuration.backend == SigningBackend.GCS:
            if configuration.gcs is None:
                raise ConfigurationError("Missing GCS signer configuration.")

            return GcsSigner(configuration=configuration.gcs)

        if configuration.backend == SigningBackend.S3:
            if configuration.s3 is None:
                raise ConfigurationError("Missing S3 signer configuration.")

            return S3Signer(configuration=configuration.s3)

        return NoopSigner()
