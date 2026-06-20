from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.signing import SigningOutcome, SigningRequest


class SigningPort(ABC):
    """
    Convert stored artifact URIs into browser-fetchable signed URLs.

    Hosts construct one signer per process via a concrete adapter and
    inject it into `ConversationService`. The service consults the port
    when rendering artifact-bearing responses so consumers never see
    unsigned object URIs by accident.
    """

    @property
    @abstractmethod
    def ttl_seconds(self) -> int:
        """
        Return the signed-URL TTL in seconds the adapter applies.
        """

        raise NotImplementedError

    @abstractmethod
    async def sign(self, *, request: SigningRequest) -> SigningOutcome:
        """
        Sign one artifact URI; preserve the original on failure.
        """

        raise NotImplementedError
