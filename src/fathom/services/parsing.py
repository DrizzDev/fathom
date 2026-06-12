from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.exceptions import VisionError
from fathom.interfaces import IResponseParser
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult
from fathom.schemas.tool_requests import ExploreUIRequest, ScreenTranslation

logger = getLogger(__name__)


class ToolResponseParser(IResponseParser):
    """
    Parses raw LLM tool call responses into domain objects.

    Exploration-only parser — handles explore_ui and optional
    describe_screen tool calls from a single LLM response.
    """

    __PRIMARY_TOOLS = {"explore_ui"}
    __SECONDARY_TOOLS = {"describe_screen"}

    def parse(self, response: Any) -> AnalysisResult:
        """
        Parses the explore_ui tool call from the LLM response.
        """

        try:
            candidates = response.candidates
            if not candidates:
                block_reason = getattr(response, "prompt_feedback", None)
                reason_str = str(block_reason) if block_reason else "unknown"
                logger.warning(f"Response blocked (no candidates): {reason_str}")
                return self.__create_fallback_result(
                    message=f"Response blocked by safety filter: {reason_str}"
                )

            candidate = candidates[0]

            _BLOCKED_REASONS = {
                "SAFETY",
                "RECITATION",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                2,
                3,
                4,
                5,
            }
            if candidate.finish_reason in _BLOCKED_REASONS:
                logger.warning(f"Response blocked by content filter: {candidate.finish_reason}")
                return self.__create_fallback_result(
                    message=f"Content filtered: {candidate.finish_reason}"
                )

            if candidate.finish_reason not in (None, "STOP", 1):
                logger.warning(f"Model failed to finish normally: {candidate.finish_reason}")

            content = candidate.content
            parts = content.parts if content and content.parts else []

            # Extract function call
            function_calls = [part.function_call for part in parts if part.function_call]

            if not function_calls:
                text_response = "".join(part.text for part in parts if part.text)
                logger.warning(f"No function call received. Text: {text_response}")
                return self.__create_fallback_result(message=text_response)

            # Find the primary (explore_ui) and optional secondary (describe_screen) calls
            primary_call = None
            describe_call = None
            for fc in function_calls:
                if fc.name in self.__PRIMARY_TOOLS:
                    primary_call = fc
                elif fc.name in self.__SECONDARY_TOOLS:
                    describe_call = fc
                else:
                    logger.warning(f"Unknown tool call ignored: {fc.name}")

            if not primary_call:
                names = [fc.name for fc in function_calls]
                logger.warning("No recognized tool calls found: %s", names)
                return self.__create_fallback_result(
                    message=f"Unrecognized tool calls: {', '.join(names)}"
                )

            # Parse the explore_ui call
            result = self.__parse_exploration(arguments=primary_call.args)
            result.metadata["tool_name"] = primary_call.name
            result.metadata["tool_args"] = dict(primary_call.args or {})

            # Extract inline rich description if the LLM also called describe_screen
            if describe_call:
                try:
                    result.metadata["rich_description"] = self.__format_translation(
                        dict(describe_call.args or {})
                    )
                except Exception:
                    logger.warning("Failed to parse describe_screen args", exc_info=True)

            return result

        except Exception as exception:
            logger.exception("Failed to parse tool response")
            raise VisionError(f"Response parsing failed: {exception}") from exception

    def __parse_exploration(self, arguments: Any) -> AnalysisResult:
        """
        Parses explore_ui tool arguments.

        Lean parser for exploration mode — no delta signals, no validation,
        no memory updates, no script export.  Just action + screen description
        + content_exhausted.
        """

        try:
            request = ExploreUIRequest.model_validate(arguments)
        except ValidationError as e:
            logger.warning(f"explore_ui validation error: {e}")
            return self.__create_fallback_result(
                message=str(arguments.get("assistant_message", ""))
            )

        data = request.action
        if not data:
            return self.__create_fallback_result(message=request.assistant_message)

        bounds = None
        tap_target = data.get("tap_target")
        if tap_target:
            bounds = Bounds(
                x=int(tap_target.get("x", 0)),
                y=int(tap_target.get("y", 0)),
                width=0,
                height=0,
                coord_system="normalized",
            )
        else:
            logger.warning(
                "No tap_target in explore_ui action — taps will fall back to screen center"
            )

        # Parse action type
        try:
            action_type = ActionType(str(data.get("action_type", "tap")).lower())
        except ValueError:
            action_type = ActionType.TAP

        target_name = data.get("target_name") or "UI Element"

        raw_region = data.get("region")
        region_value = (
            raw_region
            if raw_region
            in {"top_bar", "bottom_nav", "content", "modal", "overlay", "fab", "footer"}
            else None
        )

        raw_category = data.get("element_category")
        category_value = (
            raw_category
            if raw_category
            in {
                "global_navigation",
                "primary_action",
                "content_item",
                "filter_or_category",
                "secondary_control",
                "overlay_dismiss",
            }
            else None
        )

        raw_text = data.get("text")
        text_value = str(raw_text) if raw_text else None

        action = Action(
            bounds=bounds,
            natural_language_target=target_name,
            action_type=action_type,
            rationale=str(data.get("rationale", "")),
            confidence=self.__safe_float(data.get("confidence", 0.9), default=0.9),
            overlay_detected=bool(data.get("overlay_detected", False)),
            region=region_value,
            element_category=category_value,
            text=text_value,
        )

        # Capture exploration-specific fields in metadata
        metadata: Dict[str, Any] = {}
        if data.get("element_category"):
            metadata["element_category"] = data["element_category"]
        if data.get("expected_outcome"):
            metadata["expected_outcome"] = data["expected_outcome"]
        if data.get("region"):
            metadata["region"] = data["region"]

        return AnalysisResult(
            action=action,
            alternatives=[],
            reasoning=request.assistant_message,
            is_goal_complete=False,
            content_exhausted=request.content_exhausted or False,
            screen_description=request.screen_description or "Exploration step",
            metadata=metadata,
        )

    @staticmethod
    def __safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def __format_translation(data: Dict[str, Any]) -> str:
        """Format describe_screen tool args into the screen-description markdown."""

        return ScreenTranslation.model_validate(data).to_markdown()

    def __create_fallback_result(self, message: str, completed: bool = False) -> AnalysisResult:
        """
        Creates a generic wait result when parsing fails or no action is found.
        """

        return AnalysisResult(
            action=Action(
                confidence=0.0,
                rationale=message,
                target="No valid action",
                action_type=ActionType.WAIT,
            ),
            alternatives=[],
            reasoning=message,
            is_goal_complete=completed,
            screen_description="Fallback state",
        )
