from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Mapping, Optional, Tuple

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.constants.tools import DiagnosticSeverity, StateNamespace
from fathom.core.exceptions import ToolValidationError, VisionError
from fathom.schemas.actions import Action
from fathom.schemas.delta import DeltaSignal
from fathom.schemas.gemini_tools import (
    AskUserArgs,
    ExecuteAction,
    ExecuteUIArgs,
    RecallMemoryArgs,
    StoreMemoryArgs,
    ValidateStateArgs,
    VerifyGoalArgs,
)
from fathom.schemas.results import (
    AnalysisOutcome,
    AnalysisResult,
    GenerateResult,
    ToolErrorFeedback,
)
from fathom.schemas.tools import (
    StateUpdate,
    ToolCommand,
    ToolDiagnostic,
    ToolResponse,
)

logger = getLogger(__name__)


class ToolResponseParser:
    """
    Parses structured GenerateResult into domain objects.
    """

    # Primary tools can request executable commands; non-command tools produce response parts.
    __NON_COMMAND_TOOLS = {"store_memory", "recall_memory"}
    __PRIMARY_TOOLS = {
        "ask_user",
        "execute_ui",
        "verify_goal",
        "validate_state",
    }

    def parse(self, response: GenerateResult) -> AnalysisResult:
        """
        Parses tool calls from a GenerateResult.
        """

        try:
            # GenerateResult already has tool_calls extracted by the adapter
            function_calls = response.tool_calls

            if not function_calls:
                text_response = response.content
                logger.warning(
                    "No function call received",
                    extra={
                        "component": "core.services.parsing",
                        "event": "tool.turn.missing",
                        "tool.response.length": len(text_response or ""),
                    },
                )
                return self.__create_fallback_result(message=text_response)

            # Separate primary vs non-command tool calls.
            primary_call = None
            non_command_calls = []

            for fc in function_calls:
                name = getattr(fc, "name", "")
                if name in self.__PRIMARY_TOOLS:
                    if primary_call is None:
                        primary_call = fc
                    else:
                        logger.warning(
                            "Multiple primary tools called; ignoring extra",
                            extra={
                                "component": "core.services.parsing",
                                "event": "tool.turn.primary_ignored",
                                "tool.name": name,
                            },
                        )
                elif name in self.__NON_COMMAND_TOOLS:
                    non_command_calls.append(fc)
                else:
                    logger.warning(
                        "Unknown tool call ignored",
                        extra={
                            "component": "core.services.parsing",
                            "event": "tool.turn.unknown_ignored",
                            "tool.name": name,
                        },
                    )

            # If no primary tool, use the first non-command tool as fallback.
            if not primary_call:
                primary_call = non_command_calls.pop(0) if non_command_calls else function_calls[0]

            # Parse primary tool call
            result = self.__dispatch_parse(name=primary_call.name, arguments=primary_call.args)
            result.metadata["tool_name"] = primary_call.name
            result.metadata["tool_args"] = self.__metadata_args(
                name=primary_call.name,
                arguments=primary_call.args,
            )

            # Process non-command tool calls into the typed model-tool response envelope.
            if non_command_calls:
                tool_response = result.tool_response or ToolResponse()
                for fc in non_command_calls:
                    logger.info(
                        "Processing parallel tool call",
                        extra={
                            "component": "core.services.parsing",
                            "event": "tool.turn.parallel_call",
                            "tool.name": fc.name,
                        },
                    )
                    side_result = self.__dispatch_parse(name=fc.name, arguments=fc.args)
                    if side_result.tool_response is not None:
                        tool_response = self.__merge_tool_response(
                            primary=tool_response,
                            extra=side_result.tool_response,
                        )

                result = result.model_copy(update={"tool_response": tool_response})

            self.__log_tool_response(
                result=result,
                primary_tool=primary_call.name,
                parallel_count=len(non_command_calls),
            )
            return result

        except ToolValidationError:
            # Propagate structured validation failures so callers can retry
            # with explicit feedback instead of collapsing them into a generic error.
            raise
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

        elif name == "ask_user":
            return self.__parse_ask_user(arguments=arguments)

        else:
            raise VisionError(f"Unknown function call: {name}")

    @staticmethod
    def __validation_error(*, tool_name: str, error: ValidationError) -> ToolValidationError:
        """
        Build value-safe model feedback for malformed tool arguments.
        """

        issues = error.errors(include_input=False)
        locations = tuple(ToolResponseParser.__validation_location(issue=issue) for issue in issues)
        issue_types = tuple(str(issue.get("type", "unknown")) for issue in issues)

        logger.warning(
            "Tool schema validation failed",
            extra={
                "component": "core.services.parsing",
                "event": "tool.schema.invalid",
                "tool.name": tool_name,
                "tool.error.count": len(issues),
                "tool.error.locations": locations,
                "tool.error.types": issue_types,
            },
        )

        location_text = ", ".join(locations) if locations else "<unknown>"
        return ToolValidationError(
            ToolErrorFeedback(
                tool_name=tool_name,
                tool_call_id=None,
                error_kind="validation",
                message=(
                    f"{tool_name} arguments validation failed at {location_text}. "
                    "Fix the tool payload shape and required fields."
                ),
            )
        )

    @staticmethod
    def __validation_location(*, issue: Mapping[str, object]) -> str:
        """
        Return a dotted validation location without including input values.
        """

        location = issue.get("loc", ())
        if not location:
            return "<root>"

        if isinstance(location, (list, tuple)):
            return ".".join(str(part) for part in location)

        return str(location)

    @staticmethod
    def __merge_tool_response(
        *,
        primary: ToolResponse,
        extra: ToolResponse,
    ) -> ToolResponse:
        """
        Merge non-command tool-response parts into the primary parsed response.
        """

        if primary.command is not None and extra.command is not None:
            raise VisionError("Multiple executable tool commands emitted in one turn")

        command = primary.command or extra.command
        return ToolResponse(
            command=command,
            updates=primary.updates + extra.updates,
            data=primary.data + extra.data,
            artifacts=primary.artifacts + extra.artifacts,
            diagnostics=primary.diagnostics + extra.diagnostics,
        )

    @staticmethod
    def __updates(
        *,
        memory_updates: Optional[Dict[str, str]],
    ) -> Tuple[StateUpdate, ...]:
        """
        Convert execute_ui memory_updates into runtime updates.
        """

        if not memory_updates:
            return ()

        return tuple(
            StateUpdate(namespace=StateNamespace.MEMORY, key=key, value=value)
            for key, value in memory_updates.items()
        )

    @staticmethod
    def __metadata_args(*, name: str, arguments: Any) -> Dict[str, Any]:
        """
        Return tool metadata with memory values redacted.
        """

        metadata = dict(arguments)
        if name == "store_memory" and "value" in metadata:
            metadata["value"] = "<redacted>"

        if name == "execute_ui":
            memory_updates = metadata.get("memory_updates")
            if isinstance(memory_updates, dict):
                metadata["memory_updates"] = dict.fromkeys(memory_updates, "<redacted>")
            action = metadata.get("action")
            if isinstance(action, dict):
                metadata["action"] = ToolResponseParser.__redacted_action_metadata(action=action)

        return metadata

    @staticmethod
    def __redacted_action_metadata(*, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return execute_ui action metadata with capture values redacted.
        """

        metadata = dict(action)
        capture = metadata.get("capture")
        if isinstance(capture, dict) and "value" in capture:
            redacted_capture = dict(capture)
            redacted_capture["value"] = "<redacted>"
            metadata["capture"] = redacted_capture

        return metadata

    @staticmethod
    def __log_tool_response(
        *,
        result: AnalysisResult,
        primary_tool: str,
        parallel_count: int,
    ) -> None:
        """
        Emit a value-safe summary of the parsed model-tool turn.
        """

        response = result.tool_response or ToolResponse()
        command = response.command
        capture = command.payload.capture if command is not None else None
        logger.info(
            "Tool turn parsed",
            extra={
                "component": "core.services.parsing",
                "event": "tool.turn.parsed",
                "tool.primary": primary_tool,
                "tool.parallel.count": parallel_count,
                "command.present": command is not None,
                "command.action_type": command.action_type.value if command is not None else None,
                "update.count": len(response.updates),
                "data.count": len(response.data),
                "artifact.count": len(response.artifacts),
                "diagnostic.count": len(response.diagnostics),
                "update.namespaces": sorted(
                    {update.namespace.value for update in response.updates}
                ),
                "update.keys": sorted(update.key for update in response.updates),
                "capture.present": capture is not None,
                "capture.name": capture.name if capture is not None else None,
                "capture.value.length": len(capture.value) if capture is not None else None,
            },
        )

    @staticmethod
    def __autofill_completion_reasons(
        *,
        result: AnalysisResult,
        source_tool: str,
        rationale: Optional[str] = None,
    ) -> None:
        """
        Derive missing completion-reason fields from ``action.rationale`` so
        the completion gate's structured path remains valid even when the
        model forgets the conditionally-required fields.
        """

        reason = (
            rationale
            if rationale is not None
            else result.action.rationale
            if result.action is not None
            else None
        )
        text = (reason or "").strip()
        if not text:
            return

        if result.is_sub_goal_complete and not (result.subgoal_completion_reason or "").strip():
            object.__setattr__(result, "subgoal_completion_reason", text)
            logger.info(
                "Auto-derived missing subgoal_completion_reason from rationale",
                extra={
                    "component": "core.services.parsing",
                    "event": "parsing.completion_reason.autofilled",
                    "completion.scope": "subgoal",
                    "source.tool": source_tool,
                    "source.field": "action.rationale",
                },
            )

        if result.is_goal_complete and not (result.goal_completion_reason or "").strip():
            object.__setattr__(result, "goal_completion_reason", text)
            logger.info(
                "Auto-derived missing goal_completion_reason from rationale",
                extra={
                    "component": "core.services.parsing",
                    "event": "parsing.completion_reason.autofilled",
                    "completion.scope": "goal",
                    "source.tool": source_tool,
                    "source.field": "action.rationale",
                },
            )

    def __normalize_completion_flags(
        self,
        *,
        result: AnalysisResult,
        source_tool: str,
        raw_goal_completed: Any,
        raw_sub_goal_completed: Any,
        action_type: Optional[ActionType] = None,
        rationale: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Single source of truth for aligning completion flags with terminal COMPLETE actions.

        - Preserves raw Gemini signals in metadata for auditability.
        - Normalizes is_goal_complete / is_sub_goal_complete when action_type is COMPLETE.
        - Autofills missing completion-reason fields from ``action.rationale`` so the
          downstream completion gate can verify the claim without a fuzzy match.
        """

        # Preserve raw model emissions for debugging and analytics.
        raw_flags = {
            "goal_completed": raw_goal_completed,
            "sub_goal_completed": raw_sub_goal_completed,
        }
        result.metadata.setdefault("raw_completion_flags", raw_flags)

        # Enforce invariants for terminal COMPLETE actions BEFORE autofill so the
        # autofill check `if result.is_sub_goal_complete` sees the forced-True flag
        # and populates a missing subgoal_completion_reason from the rationale.
        effective_action_type = (
            action_type
            if action_type is not None
            else result.action.action_type
            if result.action
            else None
        )
        if effective_action_type == ActionType.COMPLETE:
            normalized = {
                "source_tool": source_tool,
                "is_sub_goal_complete": True,
                "is_goal_complete": bool(raw_goal_completed) or result.is_goal_complete,
                "reason": "COMPLETE→sub-goal complete; goal completion requires explicit signal",
            }

            if not result.is_sub_goal_complete:
                logger.warning(
                    "Inconsistent completion signals from %s: action_type=COMPLETE, "
                    "is_sub_goal_complete=%s, raw_flags=%s. Normalizing sub-goal to True.",
                    source_tool,
                    result.is_sub_goal_complete,
                    raw_flags,
                )

            object.__setattr__(result, "is_sub_goal_complete", True)
            object.__setattr__(result, "is_goal_complete", normalized["is_goal_complete"])
            result.metadata.setdefault("normalized_completion_flags", normalized)

        self.__autofill_completion_reasons(
            result=result, source_tool=source_tool, rationale=rationale
        )

        return result

    def __parse_delta_telemetry(self, args: ExecuteUIArgs) -> Optional[DeltaSignal]:
        """
        Normalize raw provider delta telemetry into a :class:`DeltaSignal`.
        Returns None when no delta-specific fields are present.
        """

        # Fast path: detect complete absence of delta signal
        if (
            args.delta_observed is None
            and args.delta_confidence is None
            and args.previous_screen_summary is None
            and args.current_screen_summary is None
            and not args.visible_anchors
            and args.top_anchor is None
            and args.bottom_anchor is None
            and (args.delta_reasoning is None or not str(args.delta_reasoning).strip())
        ):
            return None

        raw_observed = args.delta_observed
        raw_confidence = args.delta_confidence

        # Normalize confidence into [0.0, 1.0] when present.
        normalized_confidence: Optional[float] = None
        confidence_source: Optional[str] = None

        if raw_confidence is not None:
            try:
                normalized_confidence = float(raw_confidence)
                if normalized_confidence < 0.0 or normalized_confidence > 1.0:
                    normalized_confidence = max(0.0, min(1.0, normalized_confidence))
                    confidence_source = "system_clamped"
                else:
                    confidence_source = "model"
            except (TypeError, ValueError):
                # Invalid numeric payload – drop confidence but preserve raw for debugging.
                normalized_confidence = None
                confidence_source = None

        # For now, we treat delta_observed as already boolean-normalized by the schema.
        normalized_observed: Optional[bool] = raw_observed

        return DeltaSignal(
            previous_screen_summary=args.previous_screen_summary,
            current_screen_summary=args.current_screen_summary,
            delta_observed=normalized_observed,
            delta_confidence=normalized_confidence,
            delta_reasoning=args.delta_reasoning,
            raw_delta_observed=raw_observed,
            raw_delta_confidence=raw_confidence,
            confidence_source=confidence_source,
            observed_source=None,
            visible_anchors=list(args.visible_anchors or []),
            top_anchor=args.top_anchor,
            bottom_anchor=args.bottom_anchor,
        )

    def __parse_goal_verification(self, arguments: Any) -> AnalysisResult:
        """
        Parses the verify_goal tool arguments.
        """

        try:
            args = VerifyGoalArgs.model_validate(arguments or {})
        except ValidationError as error:
            raise self.__validation_error(tool_name="verify_goal", error=error) from error

        reason = args.assistant_message
        raw_goal_completed = getattr(args, "goal_completed", None)
        raw_sub_goal_completed = getattr(args, "sub_goal_completed", None)
        completed = bool(raw_goal_completed)
        sub_completed = bool(raw_sub_goal_completed)
        screen = args.current_screen

        result = AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=reason,
                target=screen,
                action_type=ActionType.COMPLETE if completed else ActionType.VALIDATE,
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=completed,
            goal_completion_reason=args.goal_completion_reason,
            is_sub_goal_complete=sub_completed,
            subgoal_completion_reason=args.subgoal_completion_reason,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            metadata={"event_type": "validation"},
            screen_description="Goal verification step",
        )

        # Normalize completion flags for terminal COMPLETE actions while preserving raw signals.
        return self.__normalize_completion_flags(
            result=result,
            source_tool="verify_goal",
            raw_goal_completed=raw_goal_completed,
            raw_sub_goal_completed=raw_sub_goal_completed,
        )

    def __parse_state_validation(self, arguments: Any) -> AnalysisResult:
        """
        Parses the validate_state tool arguments.
        """

        try:
            args = ValidateStateArgs.model_validate(arguments or {})
        except ValidationError as error:
            raise self.__validation_error(tool_name="validate_state", error=error) from error

        evidence = args.evidence
        reason = args.assistant_message
        condition = args.condition_to_verify
        raw_goal_completed = getattr(args, "goal_completed", None)
        raw_sub_goal_completed = getattr(args, "sub_goal_completed", None)
        completed = bool(raw_goal_completed)
        sub_completed = bool(raw_sub_goal_completed)

        metadata: Dict[str, Any] = {"event_type": "validation"}
        if args.condition_met is not None:
            metadata["condition_met"] = args.condition_met

        result = AnalysisResult(
            action=Action(
                confidence=1.0,
                target=condition,
                action_type=ActionType.VALIDATE,
                rationale=f"{reason} | Evidence: {evidence}",
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=completed,
            goal_completion_reason=args.goal_completion_reason,
            is_sub_goal_complete=sub_completed,
            subgoal_completion_reason=args.subgoal_completion_reason,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            metadata=metadata,
            screen_description="State validation step",
        )

        return self.__normalize_completion_flags(
            result=result,
            source_tool="validate_state",
            raw_goal_completed=raw_goal_completed,
            raw_sub_goal_completed=raw_sub_goal_completed,
        )

    def __parse_execution(self, arguments: Any) -> AnalysisResult:
        """
        Parses the execute_ui tool arguments.
        """

        try:
            args = ExecuteUIArgs.model_validate(arguments or {})
        except ValidationError as error:
            raise self.__validation_error(tool_name="execute_ui", error=error) from error

        message = args.assistant_message
        raw_goal_completed = getattr(args, "goal_completed", None)
        raw_sub_goal_completed = getattr(args, "sub_goal_completed", None)
        completed = bool(raw_goal_completed)
        sub_completed = bool(raw_sub_goal_completed)

        if args.action is None:
            return self.__create_fallback_result(message=message, completed=completed)

        data: ExecuteAction = args.action

        raw_action_type_str = (data.action_type or "").strip().lower()
        if raw_action_type_str == "enter":
            raise ToolValidationError(
                ToolErrorFeedback(
                    tool_name="execute_ui",
                    tool_call_id=None,
                    error_kind="validation",
                    message=(
                        "action_type='enter' is not a supported execute_ui action_type. "
                        "Choose a supported value from the execute_ui tool schema."
                    ),
                )
            )
        try:
            action_type = ActionType(raw_action_type_str)
        except ValueError as error:
            raise ToolValidationError(
                ToolErrorFeedback(
                    tool_name="execute_ui",
                    tool_call_id=None,
                    error_kind="validation",
                    message=(
                        "action.action_type is not a supported execute_ui action_type. "
                        "Choose a supported value from the execute_ui tool schema."
                    ),
                )
            ) from error

        logger.info(
            "Executable tool command parsed",
            extra={
                "component": "core.services.parsing",
                "event": "tool.command.parsed",
                "command.action_type": action_type.value,
                "command.label_id": data.label_id,
                "command.has_target": bool(data.target_name or data.element_name),
                "command.has_bounds": data.bbox is not None,
                "command.confidence": data.confidence,
                "capture.present": data.capture is not None,
            },
        )

        metadata_dict: Dict[str, Any] = {}
        if action_type == ActionType.VALIDATE:
            metadata_dict["event_type"] = "validation"

        parsed_delta = self.__parse_delta_telemetry(args=args)

        result = AnalysisResult(
            action=None,
            alternatives=[],
            reasoning=message,
            metadata=metadata_dict,
            tool_response=ToolResponse(
                command=ToolCommand(action_type=action_type, payload=data),
                updates=self.__updates(memory_updates=args.memory_updates),
            ),
            is_goal_complete=completed,
            goal_completion_reason=args.goal_completion_reason,
            is_sub_goal_complete=sub_completed,
            subgoal_completion_reason=args.subgoal_completion_reason,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            delta=parsed_delta,
            screen_description=message or data.rationale or "Analyzing screen...",
        )

        # Normalize completion flags for terminal COMPLETE actions while preserving raw signals.
        return self.__normalize_completion_flags(
            result=result,
            source_tool="execute_ui",
            raw_goal_completed=raw_goal_completed,
            raw_sub_goal_completed=raw_sub_goal_completed,
            action_type=action_type,
            rationale=data.rationale,
        )

    def __parse_memory_storage(self, arguments: Any) -> AnalysisResult:
        """
        Parses memory storage tool call.
        """

        try:
            args = StoreMemoryArgs.model_validate(arguments or {})
        except ValidationError as error:
            raise self.__validation_error(tool_name="store_memory", error=error) from error

        reason = args.assistant_message

        return AnalysisResult(
            action=None,
            alternatives=[],
            reasoning=reason,
            is_goal_complete=False,
            outcome=AnalysisOutcome.TOOL_RESPONSE,
            tool_response=ToolResponse(
                updates=(
                    StateUpdate(
                        namespace=StateNamespace.MEMORY,
                        key=args.key,
                        value=args.value,
                    ),
                )
            ),
            screen_description="Memory storage step",
        )

    def __parse_memory_retrieval(self, arguments: Any) -> AnalysisResult:
        """
        Parses memory retrieval tool call.
        """

        try:
            args = RecallMemoryArgs.model_validate(arguments or {})
        except ValidationError as error:
            raise self.__validation_error(tool_name="recall_memory", error=error) from error

        reason = args.assistant_message

        return AnalysisResult(
            action=None,
            alternatives=[],
            reasoning=reason,
            is_goal_complete=False,
            outcome=AnalysisOutcome.TOOL_RESPONSE,
            tool_response=ToolResponse(
                diagnostics=(
                    ToolDiagnostic(
                        severity=DiagnosticSeverity.INFO,
                        message="Memory recall requested by model tool but not read in parser.",
                        code="MEMORY_RECALL_REQUESTED",
                    ),
                ),
            ),
            screen_description="Memory retrieval step",
        )

    def __parse_ask_user(self, arguments: Any) -> AnalysisResult:
        """
        Parses ask_user tool arguments.
        """

        try:
            args = AskUserArgs.model_validate(arguments or {})
        except ValidationError as error:
            raise self.__validation_error(tool_name="ask_user", error=error) from error

        question = (args.question or "").strip()
        context = (args.context or "").strip()
        rationale = context or question or "Requesting user clarification"
        completed = bool(args.goal_completed)
        sub_completed = bool(args.sub_goal_completed)

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=rationale,
                action_type=ActionType.ASK_USER,
                target="human_assistance",
                natural_language_target="User",
                text=question or rationale,
            ),
            alternatives=[],
            reasoning=rationale,
            is_goal_complete=completed,
            goal_completion_reason=args.goal_completion_reason,
            is_sub_goal_complete=sub_completed,
            subgoal_completion_reason=args.subgoal_completion_reason,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            screen_description="User guidance requested",
            outcome=AnalysisOutcome.ASK_USER,
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
