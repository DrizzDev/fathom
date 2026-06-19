from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.ui import LabeledElement


class ScreenObservationPort(ABC):
    """
    Builds a unified screen observation from available perception sources.
    """

    @abstractmethod
    async def observe(
        self,
        *,
        capture: ScreenCapture,
        hashes: ScreenHashBundle,
        budget: PerceptionBudget,
        manifest: Tuple[LabeledElement, ...],
    ) -> ScreenObservation:
        """
        Build a normalized screen observation.
        """

        raise NotImplementedError
