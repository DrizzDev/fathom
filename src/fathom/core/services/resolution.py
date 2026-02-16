"""Reference resolution service for dynamic action parameters."""

from __future__ import annotations

from logging import getLogger
from typing import Any

from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action

logger = getLogger(__name__)


class ReferenceResolutionService:
    """Resolves references like $memory or $env in action parameters."""

    def __init__(self, ledger: MemoryPort) -> None:
        self.__ledger = ledger

    async def resolve(self, action: Action) -> Action:
        """Resolve any dynamic references in the action."""
        # Current implementation just returns the action as-is
        # but provides the hook for future reference expansion
        return action
