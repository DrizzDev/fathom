from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.qualification import QualificationVerdict


class IntentQualifierPort(ABC):
    """
    Domain contract for deciding whether an intent describes an executable UI task.

    Carries exactly one method. Resource lifecycle (LLM construction, teardown) is the composition root's
    concern and deliberately does NOT live here; see runtime/qualifier/composer.py and the
    QualifierComposition / RunnerComposition schemas.
    """

    @abstractmethod
    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Classify the intent's executability and return a structured verdict.
        """

        raise NotImplementedError
