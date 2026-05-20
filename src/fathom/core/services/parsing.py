from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.core.exceptions import ToolValidationError, VisionError
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.decisions import (
    ActDecision,
    AskUserDecision,
    ReplanDecision,
    UnactionableDecision,
)
from fathom.schemas.delta import DeltaSignal
from fathom.schemas.escape import EscapeCategory, EscapeReport
from fathom.schemas.gemini_tools import (
    AskUserArgs,
    ExecuteAction,
    ExecuteUIArgs,
    RecallMemoryArgs,
    ReportUnactionableArgs,
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

logger = getLogger(__name__)


class ToolResponseParser:
    """
    Parses structured GenerateResult into domain objects.
    """

    # Primary tools produce the AnalysisResult; side-effect tools merge into it.
    __SIDE_EFFECT_TOOLS = {"store_memory", "recall_memory"}
    __PRIMARY_TOOLS = {
        "ask_user",
        "execute_ui",
        "verify_goal",
        "validate_state",
        "request_replan",
        "report_unactionable",
    }

    @staticmethod
    def __require_bool(
        key: str,
        arguments: Any,
        tool_name: str,
        *,
        default_if_missing: Optional[bool] = None,
    ) -> bool:
        """
        Read a boolean field from model tool arguments with compatibility fallback.
        """

        if key not in arguments:
            if default_if_missing is not None:
                logger.warning(
                    "Missing mandatory Gemini completion signal '%s' in %s response; defaulting to %s",
                    key,
                    tool_name,
                    default_if_missing,
                )
                return default_if_missing
            raise VisionError(
                f"Missing mandatory Gemini completion signal '{key}' in {tool_name} response"
            )

        value = arguments.get(key)
        if isinstance(value, bool):
            return value

        if default_if_missing is not None:
            logger.warning(
                "Invalid completion signal type for '%s' in %s response; defaulting to %s",
                key,
                tool_name,
                default_if_missing,
            )
            return default_if_missing

        raise VisionError(
            f"Invalid completion signal type for '{key}' in {tool_name}: expected bool"
        )

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

        elif name == "request_replan":
            return self.__parse_request_replan(arguments=arguments)

        elif name == "report_unactionable":
            return self.__parse_report_unactionable(arguments=arguments)

        else:
            raise VisionError(f"Unknown function call: {name}")

    def __normalize_completion_flags(
        self,
        *,
        result: AnalysisResult,
        source_tool: str,
        raw_goal_completed: Any,
        raw_sub_goal_completed: Any,
    ) -> AnalysisResult:
        """
        Single source of truth for aligning completion flags with terminal COMPLETE actions.

        - Preserves raw Gemini signals in metadata for auditability.
        - Normalizes is_goal_complete / is_sub_goal_complete when action_type is COMPLETE.
        """

        # Preserve raw model emissions for debugging and analytics.
        raw_flags = {
            "goal_completed": raw_goal_completed,
            "sub_goal_completed": raw_sub_goal_completed,
        }
        result.metadata.setdefault("raw_completion_flags", raw_flags)

        # Only enforce invariants for terminal COMPLETE actions.
        if result.action.action_type == ActionType.COMPLETE:
            # In decomposed flows, COMPLETE may mean "current sub-goal complete" but not
            # necessarily "intent complete". Do not promote local completion into full
            # intent completion unless the model explicitly signals goal_completed.
            #
            # Policy:
            # - COMPLETE always implies sub-goal completion.
            # - Goal completion is respected only when explicitly signaled by the tool flags.
            normalized = {
                "source_tool": source_tool,
                "is_sub_goal_complete": True,
                "is_goal_complete": bool(raw_goal_completed) or bool(result.is_goal_complete),
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
            logger.exception("verify_goal schema validation failed: %s", error)
            feedback = ToolErrorFeedback(
                tool_name="verify_goal",
                tool_call_id=None,
                error_kind="validation",
                message=f"verify_goal arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

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
            task_status=getattr(args, "task_status", None),
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
            logger.exception("validate_state schema validation failed: %s", error)
            feedback = ToolErrorFeedback(
                tool_name="validate_state",
                tool_call_id=None,
                error_kind="validation",
                message=f"validate_state arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

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
            task_status=getattr(args, "task_status", None),
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
            logger.exception("execute_ui schema validation failed: %s", error)
            feedback = ToolErrorFeedback(
                tool_name="execute_ui",
                tool_call_id=None,
                error_kind="validation",
                message=f"execute_ui arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

        message = args.assistant_message
        raw_goal_completed = getattr(args, "goal_completed", None)
        raw_sub_goal_completed = getattr(args, "sub_goal_completed", None)
        completed = bool(raw_goal_completed)
        sub_completed = bool(raw_sub_goal_completed)

        if args.action is None:
            return self.__create_fallback_result(message=message, completed=completed)

        data: ExecuteAction = args.action

        bounds = None
        if data.bbox:
            try:
                bounds = Bounds(
                    x=data.bbox.x,
                    y=data.bbox.y,
                    width=data.bbox.width,
                    height=data.bbox.height,
                    source=CoordinateSource.MODEL,
                    coordinate_system=CoordinateSystem.from_legacy(
                        data.bbox.coordinate_system,
                    ),
                )
            except Exception:
                logger.warning("Ignoring malformed bbox payload from GeminiBBox: %s", data.bbox)

        raw_action_type_str = str(data.action_type or "").strip().lower()
        try:
            action_type = ActionType(raw_action_type_str or "wait")
        except ValueError:
            logger.warning(
                "Invalid action_type '%s' from VLM; defaulting to WAIT.", raw_action_type_str
            )
            action_type = ActionType.WAIT

        # Support variations from different prompt/model versions
        text = data.text or data.text_to_type
        wait_duration: Optional[float] = data.wait_duration

        validation_reason = data.validation_reason

        # Prefer model-provided structured targets when the primary name is generic.
        raw_target_name = data.target_name or data.element_name
        script_target = data.script_target

        resolved_target_name: Optional[str] = raw_target_name
        if Normalizer.is_generic_target_name(resolved_target_name):
            structured_fallback = None
            if script_target and not Normalizer.is_generic_target_name(script_target):
                structured_fallback = script_target

            if structured_fallback:
                logger.info(
                    "Repaired generic Gemini target '%s' using structured field '%s'",
                    raw_target_name,
                    structured_fallback,
                )
                resolved_target_name = structured_fallback
            else:
                # Do not synthesize label- or bounds-based tags; keep a simple fallback.
                resolved_target_name = raw_target_name or "element"

        condition = data.condition
        is_conditional = data.is_conditional
        conditional_type = data.conditional_type
        overlay_detected = data.overlay_detected

        target_type = data.target_type

        action = Action(
            bounds=bounds,
            target=resolved_target_name or "element",
            condition=condition,
            is_conditional=is_conditional,
            conditional_type=conditional_type,
            overlay_detected=overlay_detected,
            action_type=action_type,
            target_type=target_type,
            script_target=script_target,
            wait_duration=wait_duration,
            text=str(text) if text else None,
            validation_reason=validation_reason,
            natural_language_target=resolved_target_name or "element",
            rationale=str(data.rationale or ""),
            is_valid=bool(data.is_valid),
            confidence=float(data.confidence),
            memory_updates=args.memory_updates,
            label_id=data.label_id,
            # Structured export signals (VLM-provided, authoritative).
            export_target=data.export_target,
            scroll_target=data.scroll_target,
            wait_subject=data.wait_subject,
            wait_pattern=data.wait_pattern,
            is_app_launcher=data.is_app_launcher,
            target_is_generic=data.target_is_generic,
            target_element_type=data.target_element_type,
            validation_subject=data.validation_subject,
            validation_pattern=data.validation_pattern,
        )

        logger.info(
            "Planner tool call parsed",
            extra={
                "component": "core.services.parsing",
                "event": "planner.tool_call.parsed",
                "action.type": action_type.value,
                "action.label_id": data.label_id,
                "action.target": resolved_target_name,
                "action.has_bounds": bounds is not None,
                "action.confidence": float(data.confidence),
                "action.bounds": (
                    {
                        "x": bounds.x,
                        "y": bounds.y,
                        "width": bounds.width,
                        "height": bounds.height,
                        "system": bounds.system.value,
                    }
                    if bounds is not None
                    else None
                ),
            },
        )

        alternatives: List[Action] = []

        metadata_dict: Dict[str, Any] = {}
        if action_type == ActionType.VALIDATE:
            metadata_dict["event_type"] = "validation"

        parsed_delta = self.__parse_delta_telemetry(args=args)

        result = AnalysisResult(
            action=action,
            alternatives=alternatives,
            reasoning=message,
            metadata=metadata_dict,
            is_goal_complete=completed,
            goal_completion_reason=args.goal_completion_reason,
            is_sub_goal_complete=sub_completed,
            task_status=getattr(args, "task_status", None),
            subgoal_completion_reason=args.subgoal_completion_reason,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            delta=parsed_delta,
            screen_description=message or action.rationale or "Analyzing screen...",
            decision=ActDecision(
                action=action,
                rationale=message or action.rationale,
            ),
        )

        # Normalize completion flags for terminal COMPLETE actions while preserving raw signals.
        return self.__normalize_completion_flags(
            result=result,
            source_tool="execute_ui",
            raw_goal_completed=raw_goal_completed,
            raw_sub_goal_completed=raw_sub_goal_completed,
        )

    @staticmethod
    def __safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def __parse_memory_storage(self, arguments: Any) -> AnalysisResult:
        """
        Parses memory storage tool call.
        """

        try:
            args = StoreMemoryArgs.model_validate(arguments or {})
        except ValidationError as error:
            logger.exception("store_memory schema validation failed: %s", error)
            feedback = ToolErrorFeedback(
                tool_name="store_memory",
                tool_call_id=None,
                error_kind="validation",
                message=f"store_memory arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

        key = args.key
        value = args.value
        reason = args.assistant_message

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

        try:
            args = RecallMemoryArgs.model_validate(arguments or {})
        except ValidationError as error:
            logger.exception("recall_memory schema validation failed: %s", error)
            feedback = ToolErrorFeedback(
                tool_name="recall_memory",
                tool_call_id=None,
                error_kind="validation",
                message=f"recall_memory arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

        key = args.key
        reason = args.assistant_message

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

    def __parse_ask_user(self, arguments: Any) -> AnalysisResult:
        """
        Parses ask_user tool arguments.
        """

        try:
            args = AskUserArgs.model_validate(arguments or {})
        except ValidationError as error:
            logger.exception("ask_user schema validation failed: %s", error)
            feedback = ToolErrorFeedback(
                tool_name="ask_user",
                tool_call_id=None,
                error_kind="validation",
                message=f"ask_user arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

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
            task_status=getattr(args, "task_status", None),
            subgoal_completion_reason=args.subgoal_completion_reason,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            screen_description="User guidance requested",
            outcome=AnalysisOutcome.ASK_USER,
            decision=AskUserDecision(
                question=question or rationale,
                reason=rationale,
            ),
        )

    def __parse_request_replan(self, arguments: Any) -> AnalysisResult:
        """
        Parse the ``request_replan`` tool call into a typed
        :class:`EscapeReport` payload on the :class:`AnalysisResult`.

        Validates the typed contract at this boundary: ``category`` must
        be a known :class:`EscapeCategory` value and ``detail`` must be
        a non-empty string. Invalid arguments raise
        :class:`ToolValidationError` so the planner never sees a
        half-formed escape signal — :class:`EscapeReport` itself enforces
        the same invariants via Pydantic, and this method maps any
        validation error onto the structured tool-feedback path.

        The placeholder action is a non-spatial WAIT so existing
        downstream code that expects ``result.action`` to be valid keeps
        working; the planner branches on ``outcome`` and on
        ``escape_report.category``, not on the action itself.
        """

        raw = arguments or {}
        try:
            escape_report = EscapeReport(
                detail=str(raw.get("detail", "")).strip(),
                category=EscapeCategory(str(raw.get("category", "")).strip()),
            )
        except (ValueError, ValidationError) as error:
            logger.exception("request_replan schema validation failed: %s", error)
            raise ToolValidationError(
                feedback=ToolErrorFeedback(
                    tool_call_id=None,
                    error_kind="validation",
                    tool_name="request_replan",
                    message=f"request_replan arguments validation failed: {error}",
                )
            ) from error

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                target="request_replan",
                action_type=ActionType.WAIT,
                rationale=escape_report.detail,
            ),
            alternatives=[],
            is_goal_complete=False,
            is_sub_goal_complete=False,
            escape_report=escape_report,
            reasoning=escape_report.detail,
            outcome=AnalysisOutcome.REQUEST_REPLAN,
            screen_description=(f"Agent requested replan ({escape_report.category.value})"),
            decision=ReplanDecision(reason=escape_report.detail),
        )

    def __parse_report_unactionable(self, arguments: Any) -> AnalysisResult:
        """
        Parses report_unactionable tool arguments.
        """

        try:
            args = ReportUnactionableArgs.model_validate(arguments or {})
        except ValidationError as error:
            logger.exception("report_unactionable schema validation failed: %s", error)
            feedback = ToolErrorFeedback(
                tool_name="report_unactionable",
                tool_call_id=None,
                error_kind="validation",
                message=f"report_unactionable arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

        reason = args.reason.strip()
        escape_report = EscapeReport(
            category=EscapeCategory.PRECONDITION_NOT_MET,
            detail=reason,
        )

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                target="report_unactionable",
                action_type=ActionType.WAIT,
                rationale=reason,
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=bool(args.goal_completed),
            is_sub_goal_complete=bool(args.sub_goal_completed),
            escape_report=escape_report,
            outcome=AnalysisOutcome.REPORT_UNACTIONABLE,
            screen_description="Agent reported current screen as unactionable",
            decision=UnactionableDecision(reason=reason),
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
