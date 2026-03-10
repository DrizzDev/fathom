from __future__ import annotations

from logging import getLogger
from typing import Any, Dict, Optional

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.core.exceptions import VisionError
from fathom.core.services.normalizer import Normalizer
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.delta import GeminiDeltaSignal
from fathom.schemas.gemini_tools import (
    AskUserArgs,
    ExecuteAction,
    ExecuteUIArgs,
    RecallMemoryArgs,
    StoreMemoryArgs,
    ValidateStateArgs,
    VerifyGoalArgs,
)
from fathom.schemas.results import AnalysisResult, GenerateResult

logger = getLogger(__name__)


class ToolResponseParser:
    """
    Parses structured GenerateResult into domain objects.
    """

    # Primary tools produce the AnalysisResult; side-effect tools merge into it.
    __SIDE_EFFECT_TOOLS = {"store_memory", "recall_memory"}
    __PRIMARY_TOOLS = {"execute_ui", "verify_goal", "validate_state", "ask_user"}

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
            normalized = {
                "is_goal_complete": True,
                "is_sub_goal_complete": True,
                "source_tool": source_tool,
                "reason": "enforced COMPLETE→flags invariant",
            }

            if not result.is_goal_complete or not result.is_sub_goal_complete:
                logger.warning(
                    "Inconsistent completion signals from %s: action_type=COMPLETE, "
                    "is_goal_complete=%s, is_sub_goal_complete=%s, raw_flags=%s. "
                    "Normalizing both flags to True.",
                    source_tool,
                    result.is_goal_complete,
                    result.is_sub_goal_complete,
                    raw_flags,
                )

            object.__setattr__(result, "is_goal_complete", True)
            object.__setattr__(result, "is_sub_goal_complete", True)
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
            raise VisionError(f"verify_goal arguments validation failed: {error}") from error

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
            is_sub_goal_complete=sub_completed,
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
            raise VisionError(f"validate_state arguments validation failed: {error}") from error

        evidence = args.evidence
        reason = args.assistant_message
        condition = args.condition_to_verify
        completed = bool(args.goal_completed)
        sub_completed = bool(args.sub_goal_completed)

        return AnalysisResult(
            action=Action(
                confidence=1.0,
                target=condition,
                action_type=ActionType.VALIDATE,
                rationale=f"{reason} | Evidence: {evidence}",
            ),
            alternatives=[],
            reasoning=reason,
            is_goal_complete=completed,
            is_sub_goal_complete=sub_completed,
            metadata={"event_type": "validation"},
            screen_description="State validation step",
        )

    def __parse_execution(self, arguments: Any) -> AnalysisResult:
        """
        Parses the execute_ui tool arguments.
        """

        try:
            args = ExecuteUIArgs.model_validate(arguments or {})
        except ValidationError as error:
            logger.exception("execute_ui schema validation failed: %s", error)
            raise VisionError(f"execute_ui arguments validation failed: {error}") from error

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
                bounds = Bounds(
                    x=data.bbox.x,
                    y=data.bbox.y,
                    width=data.bbox.width,
                    height=data.bbox.height,
                    coord_system=data.bbox.coord_system,
                )
            except Exception:
                logger.warning("Ignoring malformed bbox payload from GeminiBBox: %s", data.bbox)

        try:
            action_type = ActionType(str(data.action_type or "wait").lower())
        except ValueError:
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
        )

        metadata_dict: Dict[str, Any] = {}
        if action_type == ActionType.VALIDATE:
            metadata_dict["event_type"] = "validation"

        parsed_delta = self.__parse_delta_telemetry(args=args)

        result = AnalysisResult(
            action=action,
            alternatives=[],
            reasoning=message,
            metadata=metadata_dict,
            is_goal_complete=completed,
            goal_completion_reason=args.goal_completion_reason if completed else None,
            is_sub_goal_complete=sub_completed,
            subgoal_completion_reason=args.subgoal_completion_reason if sub_completed else None,
            completion_criteria_met=args.completion_criteria_met,
            content_exhausted=bool(args.content_exhausted),
            gemini_delta=parsed_delta,
            screen_description=message or action.rationale or "Analyzing screen...",
        )

        # Normalize completion flags for terminal COMPLETE actions while preserving raw signals.
        return self.__normalize_completion_flags(
            result=result,
            source_tool="execute_ui",
            raw_goal_completed=raw_goal_completed,
            raw_sub_goal_completed=raw_sub_goal_completed,
        )

    @staticmethod
    def __safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def __safe_optional_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False

        return None

    @staticmethod
    def __safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    def __parse_bounds(self, data: Dict[str, Any]) -> Optional[Bounds]:
        serialization = data.get("bbox")

        if not isinstance(serialization, dict):
            return None

        x = self.__safe_int(serialization.get("x"), default=0)
        y = self.__safe_int(serialization.get("y"), default=0)
        width = self.__safe_int(serialization.get("width"), default=0)
        height = self.__safe_int(serialization.get("height"), default=0)

        if width <= 0 or height <= 0:
            logger.warning(
                "Ignoring invalid bbox with non-positive dimensions: width=%s height=%s",
                width,
                height,
            )
            return None

        coord_system_raw = str(serialization.get("coord_system", "normalized")).strip().lower()
        coord_system = (
            coord_system_raw if coord_system_raw in {"normalized", "pixel"} else "normalized"
        )

        try:
            return Bounds(
                x=x,
                y=y,
                width=width,
                height=height,
                coord_system=coord_system,
            )
        except Exception:
            logger.warning("Ignoring malformed bbox payload: %s", serialization)
            return None

    def __parse_memory_storage(self, arguments: Any) -> AnalysisResult:
        """
        Parses memory storage tool call.
        """

        try:
            args = StoreMemoryArgs.model_validate(arguments or {})
        except ValidationError as error:
            logger.exception("store_memory schema validation failed: %s", error)
            raise VisionError(f"store_memory arguments validation failed: {error}") from error

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
            raise VisionError(f"recall_memory arguments validation failed: {error}") from error

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
            raise VisionError(f"ask_user arguments validation failed: {error}") from error

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
            is_sub_goal_complete=sub_completed,
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
            reasoning=message,
            is_goal_complete=completed,
            screen_description=message or "Fallback state",
        )
