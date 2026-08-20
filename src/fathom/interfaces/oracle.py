from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.criterion import Verdict


class OraclePort(ABC):
    """
    Reads one pass/fail criterion against a settled screen capture.
    """

    @abstractmethod
    async def read(self, *, criterion: str, image: bytes) -> Verdict:
        """
        Return the structured verdict for one criterion against one settled capture.
        """

        raise NotImplementedError
