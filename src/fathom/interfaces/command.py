from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.command import CommandAnchor, CommandScope
from fathom.schemas.observation import ScreenObservation
from fathom.utils.coordinates import CoordinateConverter


class CommandScopeResolvePort(ABC):
    """
    Port that resolves one command anchor into one executable scope.
    """

    @abstractmethod
    async def resolve(
        self,
        *,
        anchor: CommandAnchor,
        fallback: CommandScope,
        observation: ScreenObservation,
        converter: CoordinateConverter,
    ) -> CommandScope:
        """
        Resolve one anchor into one concrete execution scope.
        """

        raise NotImplementedError
