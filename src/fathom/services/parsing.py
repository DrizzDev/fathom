from __future__ import annotations

from logging import getLogger
from typing import Any

from fathom.constants import ActionType
from fathom.exceptions import VisionError
from fathom.interfaces import IResponseParser
from fathom.schemas.actions import Action, BoundingBox
from fathom.schemas.results import AnalysisResult

logger = getLogger(__name__)


class ToolResponseParser(IResponseParser):
    """
    Parses raw LLM tool call responses into domain objects.
    """

    def parse(self, response: Any) -> AnalysisResult:
        """
        Parses the tool call from the LLM response.
        """

        try:
            candidates = response.candidates
            if not candidates:
                raise VisionError("No candidates in response")

            candidate = candidates[0]

            # Handle token limit or other non-success finish reasons
            if candidate.finish_reason not in (
                None,
                "STOP",
                1,
            ):  # 1 is often STOP in some SDK versions
                logger.warning(f"Model failed to finish normally: {candidate.finish_reason}")

            content = candidate.content
            parts = content.parts if content and content.parts else []

            function_call = None
            for part in parts:
                if part.function_call:
                    function_call = part.function_call
                    break

            if not function_call:
                text_response = "".join(part.text for part in parts if part.text)
                logger.warning(f"No function call received. Text: {text_response}")
                return self.__create_fallback_result(text_response)

            name = function_call.name
            args = function_call.args

            if name == "verify_goal_completion":
                return self.__parse_verification(args)

            if name == "execute_ui_actions":
                return self.__parse_execution(args)

            raise VisionError(f"Unknown function call: {name}")

        except Exception as exception:
            logger.exception("Failed to parse tool response")
            raise VisionError(f"Response parsing failed: {exception}") from exception

    def __parse_verification(self, args: Any) -> AnalysisResult:
        """
        Parses the verify_goal_completion tool arguments.
        """

        reason = str(args.get("assistant_message", ""))
        is_complete = bool(args.get("goal_completed", False))

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=reason,
                target="Goal Verification",
                action_type=ActionType.COMPLETE if is_complete else ActionType.WAIT,
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=is_complete,
            screen_description="Goal verification step",
        )

    def __parse_execution(self, args: Any) -> AnalysisResult:
        """
        Parses the execute_ui_actions tool arguments.
        """

        actions = args.get("actions", [])
        assistant_message = str(args.get("assistant_message", ""))
        is_complete = bool(args.get("goal_completed", False))

        if not actions:
            return self.__create_fallback_result(assistant_message, is_complete)

        # Currently processing only the first action
        bbox = None
        first_action_data = actions[0]
        bbox_data = first_action_data.get("bbox")

        if bbox_data:
            bbox = BoundingBox(
                x=int(bbox_data.get("x", 0)),
                y=int(bbox_data.get("y", 0)),
                width=int(bbox_data.get("width", 0)),
                height=int(bbox_data.get("height", 0)),
            )

        try:
            action_type = ActionType(str(first_action_data.get("action_type", "wait")).lower())
        except ValueError:
            action_type = ActionType.WAIT

        text_content = first_action_data.get("text") or first_action_data.get("text_to_type")
        wait_time = first_action_data.get("wait_duration") or first_action_data.get(
            "wait_duration_ms"
        )

        action = Action(
            bbox=bbox,
            target="UI Element",
            action_type=action_type,
            text=str(text_content) if text_content else None,
            wait_duration=int(wait_time) if wait_time else None,
            rationale=str(first_action_data.get("rationale", "")),
            confidence=float(first_action_data.get("confidence", 1.0)),
            label_id=str(first_action_data.get("label_id"))
            if first_action_data.get("label_id")
            else None,
        )

        return AnalysisResult(
            action=action,
            alternatives=[],
            is_goal_complete=is_complete,
            reasoning=assistant_message,
            screen_description="Tool-based analysis",
        )

    def __create_fallback_result(self, message: str, is_complete: bool = False) -> AnalysisResult:
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
            is_goal_complete=is_complete,
            screen_description="Fallback state",
        )
