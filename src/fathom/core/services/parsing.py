from __future__ import annotations

import contextlib
from logging import getLogger
from typing import Any, Dict, Literal, Optional, cast

from fathom.constants import ActionType
from fathom.core.exceptions import VisionError
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.results import AnalysisResult, GenerateResult

logger = getLogger(__name__)


class ToolResponseParser:
    """
    Parses structured GenerateResult into domain objects.
    """

    # Primary tools produce the AnalysisResult; side-effect tools merge into it.
    __SIDE_EFFECT_TOOLS = {"store_memory", "recall_memory"}
    __PRIMARY_TOOLS = {"execute_ui", "verify_goal", "validate_state", "ask_user"}

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
        screen = str(arguments.get("current_screen", "Goal State"))

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=reason,
                target=screen,
                action_type=ActionType.COMPLETE if completed else ActionType.VALIDATE,
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=completed,
            metadata={"event_type": "validation"},
            screen_description="Goal verification step",
        )

    def __parse_state_validation(self, arguments: Any) -> AnalysisResult:
        """
        Parses the validate_state tool arguments.
        """

        evidence = str(arguments.get("evidence", ""))
        reason = str(arguments.get("assistant_message", ""))
        condition = str(arguments.get("condition_to_verify", "State Validation"))

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                target=condition,
                action_type=ActionType.VALIDATE,
                rationale=f"{reason} | Evidence: {evidence}",
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=False,
            metadata={"event_type": "validation"},
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

        bounds = None
        data = actions[0]
        serialization = data.get("bbox")

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

        # Support variations from different prompt/model versions
        text = data.get("text") or data.get("text_to_type")
        wait_duration: Optional[float] = data.get("wait_duration")

        if wait_duration is not None:
            with contextlib.suppress(ValueError, TypeError):
                wait_duration = float(wait_duration)

        validation_reason = (
            str(data.get("validation_reason")) if data.get("validation_reason") else None
        )
        target_name = data.get("target_name") or data.get("element_name") or "UI Element"

        condition_raw = data.get("condition")
        condition = str(condition_raw).strip() if condition_raw else None

        raw_target_type = (data.get("target_type") or "").strip().lower()
        target_type: Optional[Literal["stable", "positional", "dynamic"]] = (
            cast("Literal['stable', 'positional', 'dynamic']", raw_target_type)
            if raw_target_type in ("stable", "positional", "dynamic")
            else None
        )

        script_target_raw = data.get("script_target")
        script_target = (
            str(script_target_raw).strip()
            if script_target_raw and str(script_target_raw).strip()
            else None
        )

        action = Action(
            bounds=bounds,
            target=target_name,
            condition=condition,
            action_type=action_type,
            target_type=target_type,
            script_target=script_target,
            wait_duration=wait_duration,
            text=str(text) if text else None,
            validation_reason=validation_reason,
            natural_language_target=target_name,
            rationale=str(data.get("rationale", "")),
            is_valid=bool(data.get("is_valid", True)),
            confidence=float(data.get("confidence", 1.0)),
            memory_updates=arguments.get("memory_updates"),
            label_id=str(data.get("label_id")) if data.get("label_id") else None,
        )

        metadata_dict: Dict[str, Any] = {}
        if action_type == ActionType.VALIDATE:
            metadata_dict["event_type"] = "validation"

        return AnalysisResult(
            action=action,
            alternatives=[],
            reasoning=message,
            metadata=metadata_dict,
            is_goal_complete=completed,
            screen_description=message or action.rationale or "Analyzing screen...",
        )

    def __parse_memory_storage(self, arguments: Any) -> AnalysisResult:
        """
        Parses memory storage tool call.
        """

        key = str(arguments.get("key", ""))
        value = str(arguments.get("value", ""))
        reason = str(arguments.get("assistant_message", ""))

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=reason,
                memory_updates={key: value},
                action_type=ActionType.WAIT,
                target=f"Memory Store: {key}={value}",
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

        key = str(arguments.get("key", ""))
        reason = str(arguments.get("assistant_message", ""))

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
                action_type=ActionType.WAIT,
                target="User Guidance Requested",
            ),
            alternatives=[],
            reasoning=message,
            is_goal_complete=completed,
            screen_description=message or "Fallback state",
        )
