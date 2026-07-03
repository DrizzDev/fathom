from enum import StrEnum


class SigningStatus(StrEnum):
    """
    Per-artifact signing outcome surfaced to the consumer.

    `failed` — signing raised or returned the stored URI unchanged.
    `signed` — URI was rewritten to a presigned URL with a finite TTL.
    `not_required` — URI is local or already http(s); nothing to sign.
    `unsupported` — object scheme not handled by the configured signer.
    """

    SIGNED = "signed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    NOT_REQUIRED = "not_required"


class SigningBackend(StrEnum):
    """
    Concrete artifact-signer family selected by deployment configuration.
    """

    S3 = "s3"
    GCS = "gcs"
    NOOP = "noop"
