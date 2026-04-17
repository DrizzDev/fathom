from __future__ import annotations

import os
import re
from logging import getLogger
from typing import Any, Dict, Optional, Pattern

from fathom.constants import SPATIAL_ACTION_TYPES
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action, Bounds

logger = getLogger(__name__)


class ReferenceResolutionService:
    """
    Resolves references like $memory.key or $env.VAR in action parameters.
    Also resolves UI Element Label IDs to ground-truth pixel coordinates.
    """

    def __init__(self, ledger: MemoryPort) -> None:
        """
        Initialize with memory ledger for lookups.
        """

        self.__ledger = ledger

        # Regex for variable substitution: $source.key
        self.__ref_pattern: Pattern[str] = re.compile(r"\$(memory|env)\.([a-zA-Z0-9_]+)")

        # Regex to parse bounds string: [x1,y1][x2,y2]
        self.__bounds_pattern: Pattern[str] = re.compile(r"^\[(\d+),(\d+)\]\[(\d+),(\d+)\]$")

    async def resolve(
        self,
        action: Action,
        elements: Optional[Dict[str, Any]] = None,
    ) -> Action:
        """
        Resolve dynamic references and snap to ground-truth coordinates.
        """

        # 1. Snap to Label ID (Ground Truth)
        action = self.__snap_to_label(action=action, elements=elements)

        # 2. Resolve Variable References in text fields
        updates: Dict[str, Any] = {}

        if action.text:
            updates["text"] = await self.__resolve_string(text=action.text)

        if action.target:
            updates["target"] = await self.__resolve_string(text=action.target)

        if action.natural_language_target:
            updates["natural_language_target"] = await self.__resolve_string(
                text=action.natural_language_target
            )

        if updates:
            return action.model_copy(update=updates)

        return action

    def __snap_to_label(
        self,
        action: Action,
        elements: Optional[Dict[str, Any]],
    ) -> Action:
        """
        Overwrites action bounds with ground-truth pixel coordinates if label_id matches.

        Snapping is skipped for non-spatial action types (wait, validate, complete, etc.)
        that carry no meaningful target element coordinates.
        """

        if action.action_type not in SPATIAL_ACTION_TYPES:
            return action

        if not action.label_id or not elements:
            return action

        info = elements.get(action.label_id)
        if not info:
            return action

        bounds_str = str(info.get("bounds", ""))
        if not bounds_str:
            return action

        try:
            # Parse [x1,y1][x2,y2] using pre-compiled regex for speed
            match = self.__bounds_pattern.match(bounds_str)
            if not match:
                logger.warning(f"Invalid bounds format for label {action.label_id}: {bounds_str}")
                return action

            x1, y1, x2, y2 = map(int, match.groups())
            width = x2 - x1
            height = y2 - y1

            logger.info(
                f"Snapped Action to Label [{action.label_id}] "
                f"-> Pixel Bounds: {x1},{y1} {width}x{height}"
            )

            return action.model_copy(
                update={
                    "bounds": Bounds(
                        x=x1,
                        y=y1,
                        source="xml",
                        width=width,
                        height=height,
                        coord_system="pixel",
                    ),
                }
            )

        except Exception as exception:
            logger.warning(f"Failed to snap to label {action.label_id}: {exception}")
            return action

    async def __resolve_string(self, text: str) -> str:
        """
        Resolves all $source.key matches in a string.
        """

        if not text or "$" not in text:
            return text

        # Find all matches
        matches = self.__ref_pattern.findall(text)
        if not matches:
            return text

        resolved_text = text

        for source, key in matches:
            value = await self.__fetch_value(source=source, key=key)
            if value:
                # Replace strict match $source.key
                token = f"${source}.{key}"
                resolved_text = resolved_text.replace(token, str(value))
                logger.info(f"Resolved reference '{token}' to '{value}'")
            else:
                logger.warning(f"Could not resolve reference: ${source}.{key}")

        return resolved_text

    async def __fetch_value(self, source: str, key: str) -> Optional[str]:
        """
        Fetches the value from the specified source.
        """

        if source == "env":
            return os.getenv(key)

        if source == "memory":
            return await self.__ledger.get(key=key)

        return None
