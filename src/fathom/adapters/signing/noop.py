from __future__ import annotations

from fathom.constants.signing import SigningStatus
from fathom.interfaces.signing import SigningPort
from fathom.schemas.signing import SigningOutcome, SigningRequest


class NoopSigner(SigningPort):
    """
    Pass-through signer that never rewrites URIs.

    Selected by deployments without object storage signing
    (local CLI, tests, environments where artifacts are served straight from disk or an already-public HTTP endpoint).
    """

    def __init__(self, *, ttl_seconds: int = 0) -> None:
        """
        Bind the reported TTL. Zero is the natural default because the signer never produces a presigned URL.
        """

        self.__ttl_seconds = ttl_seconds

    @property
    def ttl_seconds(self) -> int:
        """
        Return the configured TTL; always reported even though no URI is ever signed, so callers can read this in a uniform way.
        """

        return self.__ttl_seconds

    async def sign(self, *, request: SigningRequest) -> SigningOutcome:
        """
        Return the stored URI unchanged with `not_required` status.
        """

        return SigningOutcome(uri=request.uri, status=SigningStatus.NOT_REQUIRED)
