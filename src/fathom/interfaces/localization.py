from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from fathom.schemas.actions import Action
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.localization import LocalizationProposal
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture


class TargetLocalizerPort(ABC):
    """
    Single ensemble member that proposes a bounding box for a semantic target.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Stable identifier used in logs, metrics, and proposal source.
        """

        raise NotImplementedError

    @abstractmethod
    async def locate(
        self,
        *,
        action: Action,
        capture: ScreenCapture,
        budget: LocalizationBudget,
        observation: ScreenObservation,
    ) -> Optional[LocalizationProposal]:
        """
        Propose a bounding box for the action target on this screen.
        """

        raise NotImplementedError
