from __future__ import annotations

import os
import re
from logging import getLogger
from typing import Any, Dict, Optional

from fathom.interfaces import ILedger
from fathom.schemas.actions import Action

logger = getLogger(__name__)


class ReferenceResolutionService:
    """
    Resolves dynamic references in Action fields.

    Supports:
    - $memory.key -> Value from Ledger
    - $env.VAR -> Value from Environment Variables
    """

    def __init__(self, ledger: ILedger) -> None:
        self.__ledger = ledger
        self.__pattern = re.compile(r"\$(memory|env)\.([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)")

    async def resolve(self, action: Action) -> Action:
        """
        Resolves references in the action's text and target fields.
        """

        updates: Dict[str, Any] = {}

        if action.text:
            updates["text"] = await self.__resolve_string(action.text)

        if action.target:
            updates["target"] = await self.__resolve_string(action.target)

        if action.natural_language_target:
            updates["natural_language_target"] = await self.__resolve_string(
                action.natural_language_target
            )

        if updates:
            return action.model_copy(update=updates)

        return action

    async def __resolve_string(self, text: str) -> str:
        """
        Resolves all matches in a string.
        """

        if not text or "$" not in text:
            return text

        # Find all matches
        matches = self.__pattern.findall(text)
        if not matches:
            return text

        resolved_text = text
        unresolved: list[str] = []

        for source, key in matches:
            value = await self.__fetch_value(source, key)
            token = f"${source}.{key}"
            if value:
                resolved_text = resolved_text.replace(token, str(value))
            else:
                logger.warning(f"Could not resolve reference: {token}")
                resolved_text = resolved_text.replace(token, "")
                unresolved.append(token)

        if unresolved:
            logger.warning("Unresolved references replaced with empty string: %s", unresolved)

        return resolved_text

    async def __fetch_value(self, source: str, key: str) -> Optional[str]:
        """
        Fetches the value from the specified source.
        """

        if source == "env":
            return os.getenv(key)

        if source == "memory":
            return await self.__ledger.get(key)

        return None
