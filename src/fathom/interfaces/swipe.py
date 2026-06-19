from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.actions import GesturePath
from fathom.schemas.swipe import SwipeAttempt


class SwipeAttemptDispatcher(ABC):
    """
    One-attempt swipe dispatcher: one device swipe, one screenshot, one visual comparison, no retry logic.
    """

    @abstractmethod
    async def attempt(
        self,
        *,
        path: GesturePath,
        index: int,
        original_before: str,
    ) -> SwipeAttempt:
        """
        Dispatch the given gesture path and return a typed attempt result.
        """

        raise NotImplementedError
