from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from fathom.constants import ActionType, ToolName
from fathom.core.exceptions import ToolValidationError, VisionError
from fathom.schemas.actions import (
    Action,
    Bounds,
    clean_validation_subject,
    resolve_action_target,
)
from fathom.schemas.delta import GeminiDeltaSignal
from fathom.schemas.results import AnalysisResult, GenerateResult, ToolErrorFeedback
from fathom.schemas.tool_args import (
    AskUserArgs,
    ExecuteAction,
    ExecuteUIArgs,
    RecallMemoryArgs,
    StoreMemoryArgs,
    ValidateStateArgs,
    VerifyGoalArgs,
)

logger = getLogger(__name__)


class ToolResponseParser:
    """
    Parses structured GenerateResult into domain objects.
    """

    # Primary tools produce the AnalysisResult; side-effect tools merge into it.
    __SIDE_EFFECT_TOOLS: frozenset[str] = frozenset({ToolName.STORE_MEMORY, ToolName.RECALL_MEMORY})
    __PRIMARY_TOOLS: frozenset[str] = frozenset(
        {
            ToolName.EXECUTE_UI,
            ToolName.VERIFY_GOAL,
            ToolName.VALIDATE_STATE,
            ToolName.ASK_USER,
        }
    )

    @staticmethod
    def __require_bool(
        arguments: Any,
        key: str,
        tool_name: str,
        *,
        default_if_missing: Optional[bool] = None,
    ) -> bool:
        """Read a boolean field from model tool arguments with compatibility fallback."""

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
        """Route a tool call to the appropriate parser via ToolName lookup."""

        try:
            tool = ToolName(name)
        except ValueError as exc:
            raise VisionError(f"Unknown function call: {name}") from exc

        parsers = {
            ToolName.VERIFY_GOAL: self.__parse_goal_verification,
            ToolName.EXECUTE_UI: self.__parse_execution,
            ToolName.VALIDATE_STATE: self.__parse_state_validation,
            ToolName.STORE_MEMORY: self.__parse_memory_storage,
            ToolName.RECALL_MEMORY: self.__parse_memory_retrieval,
            ToolName.ASK_USER: self.__parse_ask_user,
        }
        parser = parsers.get(tool)
        if parser is None:
            raise VisionError(f"Unknown function call: {name}")
        return parser(arguments=arguments)

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

        # Coerce ToolName enum → plain string so the checkpointed metadata
        # contains only primitives. Without this, LangGraph's msgpack
        # serializer sees fathom.constants.ToolName in the state graph
        # and warns "Blocked deserialization of fathom.constants.ToolName
        # - not in allowed_msgpack_modules" on every checkpoint load.
        source_tool = str(source_tool)

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
                "is_goal_complete": bool(raw_goal_completed) or bool(result.is_goal_complete),
                "is_sub_goal_complete": True,
                "source_tool": source_tool,
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

            result.is_sub_goal_complete = True
            result.is_goal_complete = bool(normalized["is_goal_complete"])
            result.metadata.setdefault("normalized_completion_flags", normalized)

        return result

    def __parse_delta_telemetry(self, args: ExecuteUIArgs) -> Optional[GeminiDeltaSignal]:
        """
        Normalize raw Gemini delta telemetry into a GeminiDeltaSignal.

        Contract:
        - If the model provides NO delta-specific fields (all None/empty), this
          returns None so downstream code can treat it as "no signal".
        - If the model provides values, we:
            * Preserve raw values for auditability.
            * Clamp confidence into [0.0, 1.0] when out-of-range.
            * Do NOT fabricate default confidence or booleans when missing.
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

        return GeminiDeltaSignal(
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
                tool_name=ToolName.VERIFY_GOAL,
                tool_call_id=None,
                error_kind="validation",
                message=f"verify_goal arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

        reason = args.assistant_message
        evidence = getattr(args, "evidence", "") or ""
        raw_goal_completed = getattr(args, "goal_completed", None)
        raw_sub_goal_completed = getattr(args, "sub_goal_completed", None)
        completed = bool(raw_goal_completed)
        sub_completed = bool(raw_sub_goal_completed)
        screen = args.current_screen

        # Normalize to the same validate-action shape as __parse_state_validation:
        # sanitized validation_subject + merged rationale/evidence. COMPLETE
        # actions keep screen as target since the screen IS the goal state.
        validation_subject = (
            None if completed else clean_validation_subject(screen, fallback="goal state")
        )
        if evidence and reason:
            merged_rationale = f"{reason} | Evidence: {evidence}"
        elif evidence:
            merged_rationale = f"Evidence: {evidence}"
        else:
            merged_rationale = reason

        result = AnalysisResult(
            action=Action(
                confidence=1.0,
                rationale=merged_rationale,
                target=screen if completed else (validation_subject or "goal state"),
                action_type=ActionType.COMPLETE if completed else ActionType.VALIDATE,
                validation_subject=validation_subject,
            ),
            alternatives=[],
            rationale=reason,
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
            source_tool=ToolName.VERIFY_GOAL,
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
                tool_name=ToolName.VALIDATE_STATE,
                tool_call_id=None,
                error_kind="validation",
                message=f"validate_state arguments validation failed: {error}",
            )
            raise ToolValidationError(feedback) from error

        evidence = args.evidence
        reason = args.assistant_message
        raw_subject = args.validation_subject
        raw_goal_completed = getattr(args, "goal_completed", None)
        raw_sub_goal_completed = getattr(args, "sub_goal_completed", None)
        completed = bool(raw_goal_completed)
        sub_completed = bool(raw_sub_goal_completed)

        metadata: Dict[str, Any] = {"event_type": "validation"}
        if args.condition_met is not None:
            metadata["condition_met"] = args.condition_met

        sanitized_subject = clean_validation_subject(raw_subject, fallback="screen state")

        # Merge reason + evidence into the single canonical rationale
        # field. The legacy validation_reason field on ExecuteAction has
        # been removed — rationale is the one place reasoning lives.
        if evidence and reason:
            merged_rationale = f"{reason} | Evidence: {evidence}"
        elif evidence:
            merged_rationale = f"Evidence: {evidence}"
        else:
            merged_rationale = reason

        result = AnalysisResult(
            action=Action(
                confidence=1.0,
                target=sanitized_subject,
                action_type=ActionType.VALIDATE,
                rationale=merged_rationale,
                validation_subject=sanitized_subject,
            ),
            alternatives=[],
            rationale=reason,
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
            source_tool=ToolName.VALIDATE_STATE,
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
                tool_name=ToolName.EXECUTE_UI,
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

        if not args.actions:
            return self.__create_fallback_result(message=message, completed=completed)

        data: ExecuteAction = args.actions[0]

        bounds = None
        if data.bbox:
            try:
                # Treat all-zero bbox as "no bounds" (non-interactive actions
                # send {x:0, y:0, width:0, height:0} as a placeholder when bbox is required).
                if (
                    data.bbox.x == 0
                    and data.bbox.y == 0
                    and data.bbox.width == 0
                    and data.bbox.height == 0
                ):
                    bounds = None
                else:
                    bounds = Bounds(
                        x=data.bbox.x,
                        y=data.bbox.y,
                        width=data.bbox.width,
                        height=data.bbox.height,
                        coord_system=data.bbox.coord_system,
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

        # ExecuteAction._normalize_text_field has already collapsed the
        # legacy `text` alias into `text_to_type`, so downstream reads
        # only need the canonical field.
        text = data.text_to_type
        wait_duration: Optional[float] = data.wait_duration

        # ExecuteAction._normalize_target_fields has already collapsed
        # element_name / script_target into the canonical target_name and
        # derived export_target. Route through resolve_action_target so
        # every action kind reaches its canonical subject (validate →
        # validation_subject, wait → wait_subject, swipe/scroll →
        # scroll_target) instead of the historic "element" fallback.
        # ``resolve_action_target`` returns "unknown" when nothing
        # resolves; downstream consumers treat that as a placeholder
        # via ``is_resolved_target``.
        script_target = data.script_target
        resolved_target_name = resolve_action_target(
            action_type=action_type,
            target_name=data.target_name,
            export_target=data.export_target,
            validation_subject=data.validation_subject,
            wait_subject=data.wait_subject,
            scroll_target=data.scroll_target,
            label_id=data.label_id,
        )

        condition = data.condition
        is_conditional = data.is_conditional
        conditional_type = data.conditional_type
        overlay_detected = data.overlay_detected

        target_type = data.target_type

        action = Action(
            bounds=bounds,
            target=resolved_target_name,
            condition=condition,
            is_conditional=is_conditional,
            conditional_type=conditional_type,
            overlay_detected=overlay_detected,
            action_type=action_type,
            target_type=target_type,
            script_target=script_target,
            wait_duration=wait_duration,
            text=str(text) if text else None,
            natural_language_target=resolved_target_name,
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

        # Parse alternative actions from actions[1:] if provided.
        alternatives: List[Action] = []
        for alt_data in args.actions[1:]:
            try:
                alt_at_str = str(alt_data.action_type or "").strip().lower()
                alt_at = ActionType(alt_at_str) if alt_at_str else ActionType.WAIT
            except ValueError:
                alt_at = ActionType.WAIT
            alt_target = resolve_action_target(
                action_type=alt_at,
                target_name=alt_data.target_name,
                export_target=alt_data.export_target,
                validation_subject=alt_data.validation_subject,
                wait_subject=alt_data.wait_subject,
                scroll_target=alt_data.scroll_target,
                label_id=alt_data.label_id,
            )
            alternatives.append(
                Action(
                    action_type=alt_at,
                    target=alt_target,
                    natural_language_target=alt_target,
                    rationale=str(alt_data.rationale or ""),
                    confidence=float(alt_data.confidence),
                    is_valid=bool(alt_data.is_valid),
                    export_target=alt_data.export_target,
                )
            )

        metadata_dict: Dict[str, Any] = {}
        if action_type == ActionType.VALIDATE:
            metadata_dict["event_type"] = "validation"

        parsed_delta = self.__parse_delta_telemetry(args=args)

        result = AnalysisResult(
            action=action,
            alternatives=alternatives,
            rationale=message,
            metadata=metadata_dict,
            is_goal_complete=completed,
            goal_completion_reason=args.goal_completion_reason,
            is_sub_goal_complete=sub_completed,
            subgoal_completion_reason=args.subgoal_completion_reason,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            gemini_delta=parsed_delta,
            screen_description=message or action.rationale or "Analyzing screen...",
        )

        # Normalize completion flags for terminal COMPLETE actions while preserving raw signals.
        return self.__normalize_completion_flags(
            result=result,
            source_tool=ToolName.EXECUTE_UI,
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
                tool_name=ToolName.STORE_MEMORY,
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
            rationale=reason,
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
                tool_name=ToolName.RECALL_MEMORY,
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
            rationale=reason,
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
                tool_name=ToolName.ASK_USER,
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
            rationale=rationale,
            is_goal_complete=completed,
            goal_completion_reason=args.goal_completion_reason,
            is_sub_goal_complete=sub_completed,
            subgoal_completion_reason=args.subgoal_completion_reason,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            screen_description="User guidance requested",
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
            rationale=message,
            is_goal_complete=completed,
            screen_description=message or "Fallback state",
        )
