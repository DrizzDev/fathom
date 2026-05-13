from __future__ import annotations

import os
import re
from logging import getLogger
from typing import Any, Dict, Optional, Pattern, Tuple

from fathom.constants import SPATIAL_ACTION_TYPES, ActionType
from fathom.interfaces.memory import MemoryPort
from fathom.schemas.actions import Action, Bounds, InputContext
from fathom.schemas.resolution import ResolveResult

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
    ) -> ResolveResult:
        """
        Resolve dynamic references and map a named target to concrete
        manifest-grounded coordinates.

        Returns a :class:`ResolveResult` whose status reflects what
        actually happened:

        - ``RESOLVED``: snap succeeded or the action is non-spatial.
        - ``UNRESOLVED``: action is spatial and named a label that
          either doesn't exist in the manifest or carries unparseable
          bounds. The Action returned still has its variable references
          substituted so the planner can surface a useful failure
          reason.

        Variable substitution (``$memory.key``, ``$env.VAR``) always
        runs, even on the unresolved path, so the diagnostic carries
        the literal substituted target.
        """

        snapped, snap_status, snap_reason = self.__snap_to_label(
            action=action, elements=elements
        )

        substituted = await self.__substitute_references(action=snapped)

        if snap_status == "resolved":
            return ResolveResult.resolved(action=substituted)

        return ResolveResult.unresolved(
            action=substituted,
            reason=snap_reason or "named target could not be located in the manifest",
        )

    async def __substitute_references(self, *, action: Action) -> Action:
        """
        Substitute ``$memory.key`` / ``$env.VAR`` references in the
        action's text fields. Returns a fresh Action when any field
        changed; returns the original otherwise.
        """

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
    ) -> Tuple[Action, str, Optional[str]]:
        """
        Overwrite action bounds with ground-truth pixel coordinates when
        ``label_id`` matches an element in the manifest.

        Returns ``(action, status, reason)`` where ``status`` is one of:

        - ``"resolved"``: action does not require snapping (non-spatial)
          OR the snap succeeded.
        - ``"unresolved"``: action is spatial and named a target that
          could not be mapped to a manifest element. ``reason`` carries
          a short mechanical diagnostic the planner can hand to the
          recovery coordinator.
        """

        if action.action_type not in SPATIAL_ACTION_TYPES:
            return action, "resolved", None

        if not action.label_id:
            return action, "unresolved", "spatial action emitted without a label_id"

        if not elements:
            return action, "unresolved", "manifest empty; no labeled elements to snap against"

        info = elements.get(action.label_id)
        if not info:
            return action, "unresolved", (
                f"label_id '{action.label_id}' not present in current manifest"
            )

        bounds_str = str(info.get("bounds", ""))
        if not bounds_str:
            return action, "unresolved", (
                f"label_id '{action.label_id}' has no bounds; element is not snappable"
            )

        match = self.__bounds_pattern.match(bounds_str)
        if not match:
            logger.warning(
                "[Resolution] invalid bounds format for label %s: %s",
                action.label_id,
                bounds_str,
                extra={
                    "component": "resolution",
                    "event": "invalid_bounds",
                    "label_id": action.label_id,
                    "bounds": bounds_str,
                },
            )
            return action, "unresolved", (
                f"label_id '{action.label_id}' has unparseable bounds '{bounds_str}'"
            )

        try:
            x1, y1, x2, y2 = map(int, match.groups())
            width = x2 - x1
            height = y2 - y1

            logger.info(
                "[Resolution] snapped action to label [%s] bounds=%d,%d %dx%d",
                action.label_id,
                x1,
                y1,
                width,
                height,
                extra={
                    "component": "resolution",
                    "event": "snapped",
                    "label_id": action.label_id,
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                },
            )

            update: Dict[str, Any] = {
                "bounds": Bounds(
                    x=x1,
                    y=y1,
                    source="xml",
                    width=width,
                    height=height,
                    coord_system="pixel",
                ),
            }

            if (
                isinstance(info, Dict)
                and action.action_type == ActionType.TYPE
                and (input_context := self.__build_input_context(element=info))
            ):
                update["input_context"] = input_context

            return action.model_copy(update=update), "resolved", None

        except Exception as exception:
            logger.warning(
                "[Resolution] failed to snap label %s: %s",
                action.label_id,
                exception,
                extra={
                    "component": "resolution",
                    "event": "snap_error",
                    "label_id": action.label_id,
                    "error": str(exception),
                },
            )
            return action, "unresolved", (
                f"snap failed for label_id '{action.label_id}': {exception}"
            )

    @staticmethod
    def __build_input_context(*, element: Dict[str, Any]) -> Optional[InputContext]:
        """
        Build an InputContext from XML element attributes when meaningful metadata exists.

        Returns None when neither a locator nor prefilled text can be derived,
        keeping input_context absent rather than populated with empty defaults.
        """

        locator = str(element.get("resource-id", "")).strip() or None
        prefilled = ReferenceResolutionService.__prefilled_text_from_element(element=element)

        if not locator and len(prefilled) == 0:
            return None

        return InputContext(locator=locator, prefilled=prefilled, source="xml")

    @staticmethod
    def __prefilled_text_from_element(*, element: Dict[str, Any]) -> str:
        """
        Return real existing text while excluding placeholder/hint text.
        """

        text = str(element.get("text", "")).strip()

        if not text:
            return ""

        placeholder = str(
            element.get("hint", "")
            or element.get("hintText", "")
            or element.get("placeholder", "")
            or element.get("placeholderText", "")
            or ""
        ).strip()

        if placeholder and text == placeholder:
            return ""

        return text

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
