from __future__ import annotations

import os
import re
from logging import getLogger
from typing import Any, Dict, Optional, Pattern

from fathom.constants import GESTURE_ACTION_TYPES, SPATIAL_ACTION_TYPES, ActionType
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

        # Regex to parse bounds string: [x1,y1][x2,y2]. Accepts negative coordinates (off-viewport scroll containers etc.)
        # The clamp step in :meth:`__snap_to_label` is responsible for bringing them back into the viewport before they reach an executor.
        self.__bounds_pattern: Pattern[str] = re.compile(
            r"^\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]$"
        )

    async def resolve(
        self,
        action: Action,
        elements: Optional[Dict[str, Any]] = None,
    ) -> ResolveResult:
        """
        Resolve dynamic references and map a named target to concrete manifest-grounded coordinates.

        Returns a :class:`ResolveResult` whose status reflects what actually happened:

        - ``RESOLVED``: snap succeeded or the action is non-spatial.
        - ``UNRESOLVED``: action is spatial and named a label that
          either doesn't exist in the manifest or carries unparsable
          bounds. The Action returned still has its variable references
          substituted so the planner can surface a useful failure reason.

        Variable substitution (``$memory.key``, ``$env.VAR``) always
        runs, even on the unresolved path, so the diagnostic carries the literal substituted target.
        """

        attempt = self.__snap_to_label(action=action, elements=elements)
        substituted = await self.__substitute_references(action=attempt.action)

        return attempt.model_copy(update={"action": substituted})

    async def substitute(self, *, action: Action) -> Action:
        """
        Substitute dynamic references without performing target localization.
        """

        return await self.__substitute_references(action=action)

    async def __substitute_references(self, *, action: Action) -> Action:
        """
        Substitute ``$memory.key`` / ``$env.VAR`` references in the
        action's text fields. Returns a fresh Action when any field changed; returns the original otherwise.
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
    ) -> ResolveResult:
        """
        Overwrite action bounds with ground-truth pixel coordinates when
        ``label_id`` matches an element in the manifest, returning a
        typed :class:`ResolveResult` carrying the outcome.

        Non-spatial actions and successful snaps return ``RESOLVED``.
        Any other path (missing ``label_id``, label not in manifest, empty / unparsable bounds, snap exception) returns
        ``UNRESOLVED`` with a short mechanical reason the planner can hand to the recovery coordinator.
        """

        if action.action_type not in SPATIAL_ACTION_TYPES:
            # Non-spatial actions (WAIT, VALIDATE, COMPLETE, BACK, HOME,
            # HIDE_KEYBOARD, SAVE_MEMORY, RETRIEVE_MEMORY, INFER,
            # ASK_USER, UNKNOWN) carry no target bounds — snapping is a
            # no-op by design. The model occasionally attaches a stray
            # ``label_id`` for narrative purposes; strip it so the
            # executor doesn't read it as a target and downstream
            # telemetry shows a clean "non-spatial" trace.
            cleaned = (
                action.model_copy(update={"label_id": None, "bounds": None})
                if action.label_id or action.bounds
                else action
            )
            if cleaned is not action:
                logger.info(
                    "[Resolution] non-spatial action passed through; stripped stray "
                    "label_id/bounds",
                    extra={
                        "component": "resolution",
                        "event": "non_spatial.pass_through",
                        "action_type": action.action_type.value,
                        "had_label_id": bool(action.label_id),
                        "had_bounds": bool(action.bounds),
                    },
                )
            return ResolveResult.resolved(action=cleaned)

        if not action.label_id:
            if action.action_type in GESTURE_ACTION_TYPES:
                return ResolveResult.resolved(action=action)

            return ResolveResult.unresolved(
                action=action,
                reason=(
                    "spatial action emitted without a label_id; model coordinates require "
                    "perception-backed localization"
                ),
            )

        if not elements:
            return ResolveResult.unresolved(
                action=action,
                reason="manifest empty; no labeled elements to snap against",
            )

        info = elements.get(action.label_id)
        if not info:
            return ResolveResult.unresolved(
                action=action,
                reason=f"label_id '{action.label_id}' not present in current manifest",
            )

        if (
            action.bounds is not None
            and action.bounds.source == "model"
            and str(info.get("source", "")).lower() != "cv"
            and not self.__element_has_semantic_descriptor(element=info)
            and action.action_type in {ActionType.TAP, ActionType.LONG_PRESS}
        ):
            return ResolveResult.unresolved(
                action=action,
                reason=(
                    f"label_id '{action.label_id}' is a generic visual container; "
                    "perception-backed localization is required"
                ),
            )

        bounds_str = str(info.get("bounds", ""))
        if not bounds_str:
            return ResolveResult.unresolved(
                action=action,
                reason=(f"label_id '{action.label_id}' has no bounds; element is not snappable"),
            )

        match = self.__bounds_pattern.match(bounds_str)
        if not match:
            logger.warning(
                "[Resolution] invalid bounds format for label %s: %s",
                action.label_id,
                bounds_str,
                extra={
                    "bounds": bounds_str,
                    "component": "resolution",
                    "event": "invalid.bounds",
                    "label_id": action.label_id,
                },
            )

            return ResolveResult.unresolved(
                action=action,
                reason=(f"label_id '{action.label_id}' has unparsable bounds '{bounds_str}'"),
            )

        try:
            raw_x1, raw_y1, raw_x2, raw_y2 = (int(value) for value in match.groups())
            x1 = max(0, raw_x1)
            y1 = max(0, raw_y1)
            x2 = max(0, raw_x2)
            y2 = max(0, raw_y2)

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                return ResolveResult.unresolved(
                    action=action,
                    reason=(
                        f"label_id '{action.label_id}' bounds collapse to zero "
                        f"after viewport clamp (raw={bounds_str})"
                    ),
                )

            logger.info(
                "[Resolution] snapped action to label [%s] bounds=%d,%d %dx%d",
                action.label_id,
                x1,
                y1,
                width,
                height,
                extra={
                    "x": x1,
                    "y": y1,
                    "width": width,
                    "height": height,
                    "event": "snapped",
                    "component": "resolution",
                    "raw_bounds": bounds_str,
                    "label_id": action.label_id,
                    "clamped": raw_x1 < 0 or raw_y1 < 0 or raw_x2 < 0 or raw_y2 < 0,
                },
            )

            update: Dict[str, Any] = {
                "bounds": Bounds(
                    x=x1,
                    y=y1,
                    source="xml",
                    width=width,
                    height=height,
                    coordinate_system="pixel",
                ),
            }

            if (
                isinstance(info, Dict)
                and action.action_type == ActionType.TYPE
                and (input_context := self.__build_input_context(element=info))
            ):
                update["input_context"] = input_context

            return ResolveResult.resolved(action=action.model_copy(update=update))

        except Exception as exception:
            logger.warning(
                "[Resolution] failed to snap label %s: %s",
                action.label_id,
                exception,
                extra={
                    "event": "snap.error",
                    "error": str(exception),
                    "component": "resolution",
                    "label_id": action.label_id,
                },
            )
            return ResolveResult.unresolved(
                action=action,
                reason=f"snap failed for label_id '{action.label_id}': {exception}",
            )

    @staticmethod
    def __element_has_semantic_descriptor(*, element: Dict[str, Any]) -> bool:
        """
        Return whether a manifest element carries enough semantic text to be trusted over a model-provided visual bbox.
        """

        descriptor_keys = (
            "text",
            "name",
            "label",
            "hint",
            "value",
            "resource-id",
            "content-desc",
            "accessibility_label",
            "accessibility-label",
        )

        for key in descriptor_keys:
            value = element.get(key)
            if value is not None and str(value).strip():
                return True

        return False

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
