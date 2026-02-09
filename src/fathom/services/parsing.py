from __future__ import annotations

from logging import getLogger
from typing import Any

from fathom.constants import ActionType
from fathom.exceptions import VisionError
from fathom.interfaces import IResponseParser
from fathom.schemas.actions import Action, Bounds
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
                return self.__create_fallback_result(message=text_response)

            result: AnalysisResult
            name = function_call.name
            arguments = function_call.args

            if name == "verify_goal":
                result = self.__parse_goal_verification(arguments=arguments)

            elif name == "execute_ui":
                result = self.__parse_execution(arguments=arguments)

            elif name == "validate_state":
                result = self.__parse_state_validation(arguments=arguments)

            elif name == "store_memory":
                result = self.__parse_memory_storage(arguments=arguments)

            elif name == "recall_memory":
                result = self.__parse_memory_retrieval(arguments=arguments)

            else:
                raise VisionError(f"Unknown function call: {name}")

            # Inject raw tool metadata for UI rendering later
            result.metadata["tool_name"] = name
            result.metadata["tool_args"] = dict(arguments)

            return result

        except Exception as exception:
            logger.exception("Failed to parse tool response")
            raise VisionError(f"Response parsing failed: {exception}") from exception

    def __parse_goal_verification(self, arguments: Any) -> AnalysisResult:
        """
        Parses the verify_goal tool arguments.
        """

        reason = str(arguments.get("assistant_message", ""))
        completed = bool(arguments.get("goal_completed", False))

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=reason,
                target="Goal Verification",
                action_type=ActionType.COMPLETE if completed else ActionType.WAIT,
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=completed,
            screen_description="Goal verification step",
        )

    def __parse_state_validation(self, arguments: Any) -> AnalysisResult:
        """
        Parses the validate_state tool arguments.
        """

        evidence = str(arguments.get("evidence", ""))
        reason = str(arguments.get("assistant_message", ""))

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                target="State Validation",
                action_type=ActionType.WAIT,  # Validation alone doesn't complete goals
                rationale=f"{reason} | Evidence: {evidence}",
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=False,
            screen_description="State validation step",
        )

    def __parse_execution(self, arguments: Any) -> AnalysisResult:
        """
        Parses the execute_ui tool arguments.
        """

        actions = arguments.get("actions", [])
        message = str(arguments.get("assistant_message", ""))
        completed = bool(arguments.get("goal_completed", False))

        if not actions:
            return self.__create_fallback_result(message=message, completed=completed)

        # Currently processing only the first action
        bounds = None
        data = actions[0]
        serialization = data.get("bbox")

        if serialization:
            bounds = Bounds(
                x=int(serialization.get("x", 0)),
                y=int(serialization.get("y", 0)),
                width=int(serialization.get("width", 0)),
                height=int(serialization.get("height", 0)),
            )

        try:
            action_type = ActionType(str(data.get("action_type", "wait")).lower())
        except ValueError:
            action_type = ActionType.WAIT

        updates = arguments.get("memory_updates")
        text = data.get("text") or data.get("text_to_type")
        wait = data.get("wait_duration") or data.get("wait_duration_ms")

        action = Action(
            bounds=bounds,
            target="UI Element",
            memory_updates=updates,
            action_type=action_type,
            text=str(text) if text else None,
            wait_duration=int(wait) if wait else None,
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 1.0)),
            label_id=str(data.get("label_id")) if data.get("label_id") else None,
        )

        return AnalysisResult(
            action=action,
            alternatives=[],
            reasoning=message,
            is_goal_complete=completed,
            screen_description="Tool-based analysis",
        )

    def __parse_memory_storage(self, arguments: Any) -> AnalysisResult:
        """
        Parses memory storage tool call.
        """

        reason = str(arguments.get("assistant_message", ""))
        key = str(arguments.get("key", ""))
        value = str(arguments.get("value", ""))

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=reason,
                memory_updates={key: value},
                target=f"Memory Store: {key}={value}",
                action_type=ActionType.WAIT,  # Placeholder, will be handled by strategy
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=False,
            screen_description="Memory storage step",
        )

    def __parse_memory_retrieval(self, arguments: Any) -> AnalysisResult:
        """
        Parses memory retrieval tool call.
        """

        reason = str(arguments.get("assistant_message", ""))
        key = str(arguments.get("key", ""))

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=reason,
                action_type=ActionType.WAIT,
                target=f"Memory Recall: {key}",
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=False,
            screen_description="Memory retrieval step",
        )

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
