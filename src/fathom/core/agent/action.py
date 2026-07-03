from __future__ import annotations

from typing import Optional

from fathom.constants import ActionType
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.gemini_tools import ExecuteAction
from fathom.schemas.tools import AcceptedCommand


class ActionBuilder:
    """
    Builds executable actions from catalog-accepted tool commands.
    """

    def build(self, *, command: AcceptedCommand) -> Action:
        """
        Materialize one deterministic Action from an accepted execute_ui command.
        """

        data = command.payload
        target = self.__target(data=data, action_type=command.action_type)
        surface = data.surface
        if surface is None and self.__is_scroll(action_type=command.action_type):
            surface = target

        text = data.text or data.text_to_type
        return Action(
            bounds=self.__bounds(data=data),
            target=target,
            condition=data.condition,
            is_conditional=data.is_conditional,
            conditional_type=data.conditional_type,
            overlay_detected=data.overlay_detected,
            action_type=command.action_type,
            target_type=data.target_type,
            script_target=data.script_target,
            surface=surface,
            wait_duration=data.wait_duration,
            text=str(text) if text else None,
            validation_reason=data.validation_reason,
            natural_language_target=target,
            rationale=str(data.rationale or ""),
            is_valid=bool(data.is_valid),
            confidence=data.confidence,
            label_id=data.label_id,
            export_target=data.export_target,
            scroll_target=data.scroll_target,
            wait_subject=data.wait_subject,
            wait_pattern=data.wait_pattern,
            is_app_launcher=data.is_app_launcher,
            target_is_generic=data.target_is_generic,
            target_element_type=data.target_element_type,
            validation_subject=data.validation_subject,
            validation_pattern=data.validation_pattern,
            capture=data.capture,
        )

    @classmethod
    def __bounds(cls, *, data: ExecuteAction) -> Optional[Bounds]:
        """
        Convert optional model bounds into the internal coordinate model.
        """

        if data.bbox is None:
            return None

        bounds = Bounds(
            x=data.bbox.x,
            y=data.bbox.y,
            width=data.bbox.width,
            height=data.bbox.height,
            source=CoordinateSource.MODEL,
            coordinate_system=CoordinateSystem.from_legacy(data.bbox.coordinate_system),
        )
        if bounds.has_normalized_extent_violation():
            return bounds.model_copy(update={"system": CoordinateSystem.LOGICAL})

        return bounds

    @classmethod
    def __target(cls, *, data: ExecuteAction, action_type: ActionType) -> str:
        """
        Resolve the executable target string from structured payload fields.
        """

        target = cls.__target_name(data=data)
        if (
            cls.__is_scroll(action_type=action_type)
            and data.scroll_target
            and target == data.scroll_target
        ):
            target = None

        if target is None and cls.__is_scroll(action_type=action_type):
            target = "main scrollable area"

        return target or data.wait_subject or "unknown_target"

    @staticmethod
    def __target_name(*, data: ExecuteAction) -> Optional[str]:
        """
        Prefer structured target fields when the primary model target is generic.
        """

        target = data.target_name or data.element_name
        if not Normalizer.is_generic_target_name(target):
            return target

        for candidate in (
            data.script_target,
            data.wait_subject,
            data.validation_subject,
            data.export_target,
        ):
            if candidate and not Normalizer.is_generic_target_name(candidate):
                return candidate

        return target

    @staticmethod
    def __is_scroll(*, action_type: ActionType) -> bool:
        """
        Return whether the command is a scroll or swipe gesture.
        """

        return action_type in {
            ActionType.SWIPE_UP,
            ActionType.SWIPE_DOWN,
            ActionType.SWIPE_LEFT,
            ActionType.SWIPE_RIGHT,
            ActionType.SCROLL,
        }
