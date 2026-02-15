"""
Adapter to make MemoryPort compatible with IMemoryProvider interface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action
from fathom.schemas.screens import ScreenState


class MemoryProviderAdapter:
    """
    Adapter that makes MemoryPort compatible with IMemoryProvider.
    
    This allows old agent components to work with the new MemoryPort interface.
    """

    def __init__(self, memory: MemoryPort) -> None:
        """
        Initialize adapter with memory port.
        
        Args:
            memory: Memory port to wrap
        """
        self.__memory = memory

    async def get_all_knowledge(self) -> Dict[str, Any]:
        """Get all stored knowledge."""
        # MemoryPort doesn't have this method, return empty dict
        # The actual implementation in SQLiteMemory has this method
        if hasattr(self.__memory, 'get_all_knowledge'):
            return await self.__memory.get_all_knowledge()  # type: ignore
        return {}

    async def retrieve_knowledge(self, visual_hash: str) -> Dict[str, Any]:
        """Retrieve knowledge for a specific screen."""
        return await self.__memory.retrieve_knowledge(visual_hash=visual_hash)

    async def store_experience(
        self, visual_hash: str, action: Action, success: bool
    ) -> None:
        """Store action outcome."""
        await self.__memory.store_experience(
            visual_hash=visual_hash,
            action=action,
            success=success,
        )

    async def store_observation(
        self, screen: ScreenState, description: Optional[str]
    ) -> None:
        """Store screen observation."""
        await self.__memory.store_observation(screen=screen, description=description)
