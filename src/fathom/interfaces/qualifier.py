from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.qualification import QualificationVerdict


class IntentQualifierPort(ABC):
    """
    Abstract interface for deciding whether an intent describes an executable UI task.
    """

    @abstractmethod
    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Classify the intent's executability and return a structured verdict.
        """

        raise NotImplementedError
