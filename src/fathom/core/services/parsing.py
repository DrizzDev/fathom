"""Parser for LLM tool responses."""

from __future__ import annotations

from logging import getLogger
from typing import Any, Dict

from fathom.constants import ActionType
from fathom.exceptions import VisionError
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult, GenerateResult

logger = getLogger(__name__)


class ToolResponseParser:
    """
    Parses structured GenerateResult into domain objects.
    """

    # Primary tools produce the AnalysisResult; side-effect tools merge into it.
    __PRIMARY_TOOLS = {"execute_ui", "verify_goal", "validate_state"}
    __SIDE_EFFECT_TOOLS = {"store_memory", "recall_memory"}

    def parse(self, response: GenerateResult) -> AnalysisResult:
        """
        Parses tool calls from a GenerateResult.
        """
        try:
            # GenerateResult already has tool_calls extracted by the adapter
            function_calls = response.tool_calls

            if not function_calls:
                text_response = response.content
                logger.warning(f"No function call received. Text: {text_response}")
                return self.__create_fallback_result(message=text_response)

            # Separate primary vs side-effect tool calls
            primary_call = None
            side_effects = []

            for fc in function_calls:
                name = getattr(fc, "name", "")
                if name in self.__PRIMARY_TOOLS:
                    if primary_call is None:
                        primary_call = fc
                    else:
                        logger.warning(f"Multiple primary tools called; ignoring extra: {name}")
                elif name in self.__SIDE_EFFECT_TOOLS:
                    side_effects.append(fc)
                else:
                    logger.warning(f"Unknown tool call ignored: {name}")

            # If no primary tool, use the first side-effect as fallback
            if not primary_call:
                primary_call = side_effects.pop(0) if side_effects else function_calls[0]

            # Parse primary tool call
            result = self.__dispatch_parse(name=primary_call.name, arguments=primary_call.args)
            result.metadata["tool_name"] = primary_call.name
            result.metadata["tool_args"] = dict(primary_call.args)

            # Process side-effect tool calls (merge memory updates)
            if side_effects:
                merged_memory = dict(result.action.memory_updates or {})
                for fc in side_effects:
                    logger.info(f"Processing parallel tool call: {fc.name}")
                    side_result = self.__dispatch_parse(name=fc.name, arguments=fc.args)
                    if side_result.action.memory_updates:
                        merged_memory.update(side_result.action.memory_updates)

                if merged_memory:
                    action = result.action.model_copy(update={"memory_updates": merged_memory})
                    result = result.model_copy(update={"action": action})

            return result

        except Exception as exception:
            logger.exception("Failed to parse tool response")
            raise VisionError(f"Response parsing failed: {exception}") from exception

    def __dispatch_parse(self, name: str, arguments: Any) -> AnalysisResult:
        """
        Routes a tool call to the appropriate parser.
        """
        if name == "verify_goal":
            return self.__parse_goal_verification(arguments=arguments)
        elif name == "execute_ui":
            return self.__parse_execution(arguments=arguments)
        elif name == "validate_state":
            return self.__parse_state_validation(arguments=arguments)
        elif name == "store_memory":
            return self.__parse_memory_storage(arguments=arguments)
        elif name == "recall_memory":
            return self.__parse_memory_retrieval(arguments=arguments)
        else:
            raise VisionError(f"Unknown function call: {name}")

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
                action_type=ActionType.WAIT,
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

        data = actions[0]
        serialization = data.get("bbox")
        bounds = None

        if serialization:
            bounds = Bounds(
                x=int(serialization.get("x", 0)),
                y=int(serialization.get("y", 0)),
                width=int(serialization.get("width", 0)),
                height=int(serialization.get("height", 0)),
                coord_system=serialization.get("coord_system", "normalized"),
            )

        try:
            action_type = ActionType(str(data.get("action_type", "wait")).lower())
        except ValueError:
            action_type = ActionType.WAIT

        target_name = data.get("target_name") or data.get("element_name") or "UI Element"

        action = Action(
            bounds=bounds,
            target=target_name,
            natural_language_target=target_name,
            memory_updates=arguments.get("memory_updates"),
            action_type=action_type,
            text=str(data.get("text")) if data.get("text") else None,
            wait_duration=int(data.get("wait_duration")) if data.get("wait_duration") else None,
            rationale=str(data.get("rationale", "")),
            confidence=float(data.get("confidence", 1.0)),
            label_id=str(data.get("label_id")) if data.get("label_id") else None,
            is_valid=bool(data.get("is_valid", True)),
            validation_reason=str(data.get("validation_reason")) if data.get("validation_reason") else None,
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
                action_type=ActionType.WAIT,
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
