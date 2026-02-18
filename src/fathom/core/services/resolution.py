from __future__ import annotations

import re
from logging import getLogger

from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action

logger = getLogger(__name__)


class ReferenceResolutionService:
    """
    Resolves references like $memory_key or $env_var in action parameters.
    """

    def __init__(self, ledger: MemoryPort) -> None:
        """
        Initialize with memory ledger for lookups.
        """

        self.__ledger = ledger
        # Regex to find $var_name pattern
        self.__ref_pattern = re.compile(r"^\$([a-zA-Z0-9_]+)$")

    async def resolve(self, action: Action) -> Action:
        """
        Resolve any dynamic references in the action fields.
        Returns a new Action instance with resolved values.
        """

        # We only resolve 'text' field for now as it's the primary input vector
        if not action.text or not action.text.startswith("$"):
            return action

        resolved_text = await self.__resolve_value(action.text)

        if resolved_text != action.text:
            logger.info(f"Resolved reference '{action.text}' to '{resolved_text}'")
            # Return new copy with resolved text
            return action.model_copy(update={"text": resolved_text})

        return action

    async def __resolve_value(self, value: str) -> str:
        """
        Helper to resolve a single string value.
        """

        match = self.__ref_pattern.match(value)
        if not match:
            return value

        key = match.group(1)

        # 1. Check Memory Ledger
        if memory_val := await self.__ledger.get(key=key):
            return memory_val

        logger.warning(f"Could not resolve reference: {value}")
        return value
