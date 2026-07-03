from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from fathom.constants import ActionType
from fathom.schemas.capture import CaptureRequest

CoordSystem = Literal["normalized", "pixel"]
ConditionalType = Literal["blocker", "transient", "error", "optional"]
TargetType = Literal["stable", "positional", "dynamic"]
WaitPattern = Literal["ad", "splash", "load", "search", "generic"]
TargetElementType = Literal["button", "icon", "option", "link", "field", "text", "checkbox"]
ValidationPattern = Literal["blocker", "transient", "error", "generic"]


class GeminiBBox(BaseModel):
    """
    Lightweight bbox schema used at the tool boundary.

    Downstream we map this into the core Bounds model in actions.py.
    """

    x: int = Field(0, description="Top-left X coordinate")
    y: int = Field(0, description="Top-left Y coordinate")
    width: int = Field(..., gt=0, description="Bounding box width")
    height: int = Field(..., gt=0, description="Bounding box height")
    coordinate_system: CoordSystem = Field(
        "normalized",
        description="Coordinate system for bbox (normalized or pixel)",
    )


class GeminiDeltaTelemetry(BaseModel):
    """
    Semantic delta signal schema used by Gemini tools at the raw tool boundary.

    This model:
    - Preserves raw provider values (including the case where the model omits
      delta fields entirely).
    - Performs only light type coercion (e.g., parsing floats, normalizing
      anchor lists).
    - Defers all semantic normalization (e.g., defaulting, score clamping) to
      the core parser layer so that downstream code can distinguish:
        * "Model says delta"          (delta_observed is True/False)
        * "Model is unsure/low conf"  (delta_confidence in [0, 1] but small)
        * "Model said nothing"        (both fields are None)
    """

    previous_screen_summary: Optional[str] = None
    current_screen_summary: Optional[str] = None
    delta_observed: Optional[bool] = None
    delta_reasoning: Optional[str] = None
    delta_confidence: Optional[float] = None
    visible_anchors: List[str] = Field(default_factory=list)
    top_anchor: Optional[str] = None
    bottom_anchor: Optional[str] = None

    @field_validator("delta_confidence", mode="before")
    @classmethod
    def _parse_conf(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("visible_anchors", mode="before")
    @classmethod
    def _coerce_anchors(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]


class GeminiCompletionFlags(BaseModel):
    """
    Shared completion flags for primary Gemini tools.
    """

    goal_completed: bool = Field(
        ...,
        description="Whether the overall user goal is complete.",
    )
    sub_goal_completed: bool = Field(
        ...,
        description="Whether the current sub-goal is complete.",
    )
    goal_completion_reason: Optional[str] = None
    subgoal_completion_reason: Optional[str] = None
    completion_criteria_met: Optional[Any] = None
    content_exhausted: bool = False


class VerifyGoalArgs(GeminiCompletionFlags):
    """
    Schema for the verify_goal tool.
    """

    assistant_message: str = Field(
        "",
        description="Natural language reasoning and justification.",
    )
    current_screen: str = Field(
        "Goal State",
        description="Current screen description or identifier.",
    )


class ValidateStateArgs(GeminiCompletionFlags):
    """
    Schema for the validate_state tool.
    """

    assistant_message: str = Field(
        "",
        description="Natural language reasoning and justification.",
    )
    evidence: str = Field(
        "",
        description="Evidence from the UI or state for the validation.",
    )
    condition_to_verify: str = Field(
        "State Validation",
        description="Condition being validated.",
    )
    condition_met: Optional[bool] = None


_SWIPE_SCROLL_TYPES = frozenset(
    {
        "swipe_up",
        "swipe_down",
        "swipe_left",
        "swipe_right",
        "scroll",
    }
)
_GENERIC_EXPORT_TARGETS = frozenset(
    {
        "element",
        "ui element",
        "none",
        "label",
        "unknown",
        "a visible item",
    }
)


class ExecuteAction(BaseModel):
    """
    Single low-level UI action emitted by execute_ui.
    """

    bbox: Optional[GeminiBBox] = Field(
        default=None,
        description="Optional target bounds emitted by the planner.",
    )
    action_type: str = Field(
        default="wait",
        description="Low-level action type; mapped to internal ActionType.",
    )
    target_name: Optional[str] = Field(default=None, description="Primary human-readable target.")
    element_name: Optional[str] = Field(
        default=None,
        description="Secondary target name from older prompt variants.",
    )
    text: Optional[str] = Field(default=None, description="Text payload for type actions.")
    text_to_type: Optional[str] = Field(
        default=None,
        description="Legacy alias for text payloads used by older prompts.",
    )
    wait_duration: Optional[float] = Field(
        default=None,
        description="Requested wait duration in seconds.",
    )
    validation_reason: Optional[str] = Field(
        default=None,
        description="Structured validation rationale for validate-like actions.",
    )
    capture: Optional[CaptureRequest] = Field(
        default=None,
        description="For action_type='store' only: the semantic capture request (name, subject, value).",
    )

    condition: Optional[str] = Field(
        default=None,
        description="Guard condition text for conditional actions.",
    )
    is_conditional: bool = Field(
        default=False,
        description="Whether the action is gated by a condition.",
    )
    conditional_type: Optional[ConditionalType] = Field(
        default=None,
        description="Classification of the action guard when conditional.",
    )
    overlay_detected: bool = Field(
        default=False,
        description="Whether the model explicitly observed an overlay.",
    )

    target_type: Optional[TargetType] = Field(
        default=None,
        description="Planner hint describing target stability.",
    )
    script_target: Optional[str] = Field(
        default=None,
        description="Structured script-friendly target string when available.",
    )
    surface: Optional[str] = Field(
        default=None,
        description="Specific section, container, or on-screen area this action belongs to.",
    )

    # Structured export signals — authoritative; no heuristic fallback.
    export_target: Optional[str] = Field(
        default=None,
        description="Specific export target name when script generation needs one.",
    )
    scroll_target: Optional[str] = Field(
        default=None,
        description="Specific target or section the scroll action is trying to reach.",
    )
    wait_subject: Optional[str] = Field(
        default=None,
        description="Object or condition the wait action is waiting on.",
    )
    wait_pattern: Optional[WaitPattern] = Field(
        default=None,
        description="Structured wait pattern classification.",
    )
    is_app_launcher: bool = Field(
        default=False,
        description="Whether the target launches an app from a launcher surface.",
    )
    target_is_generic: Optional[bool] = Field(
        default=None,
        description="Whether the planner considers the target name generic.",
    )
    target_element_type: Optional[TargetElementType] = Field(
        default=None,
        description="Structured target element type emitted by the planner.",
    )
    validation_subject: Optional[str] = Field(
        default=None,
        description="Primary subject of a validate action.",
    )
    validation_pattern: Optional[ValidationPattern] = Field(
        default=None,
        description="Structured validation pattern emitted by the planner.",
    )

    rationale: Optional[str] = Field(default=None, description="Planner rationale for the action.")
    is_valid: bool = Field(
        default=True, description="Whether the planner considered the action valid."
    )
    confidence: float = Field(
        description="Required planner confidence in [0, 1].",
    )
    label_id: Optional[str] = Field(
        default=None,
        description="Resolved manifest label identifier when the planner grounded to one.",
    )

    @field_validator("wait_duration", mode="before")
    @classmethod
    def __parse_wait_duration(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("confidence", mode="before")
    @classmethod
    def __parse_confidence(cls, value: Any) -> float:
        """
        Parse planner confidence into a bounded float.
        """

        try:
            confidence = float(value)
        except (TypeError, ValueError) as exception:
            raise ValueError("confidence must be a numeric value in [0, 1].") from exception

        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be in [0, 1].")

        return confidence

    @field_validator("label_id", mode="before")
    @classmethod
    def __to_str_label(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    @field_validator("export_target", mode="before")
    @classmethod
    def __reject_generic_export_target(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.lower() in _GENERIC_EXPORT_TARGETS:
            raise ValueError(
                f"export_target must be specific, not a generic placeholder like '{text}'. "
                "Use the actual element name visible on screen."
            )
        return text

    @model_validator(mode="after")
    def __enforce_structured_signals(self) -> "ExecuteAction":
        at = (self.action_type or "").strip().lower()

        # scroll_target is required for swipe/scroll actions.
        if at in _SWIPE_SCROLL_TYPES and not (self.scroll_target or "").strip():
            raise ValueError(
                f"scroll_target is required for action_type='{at}'. "
                "Provide the element or section being scrolled to find."
            )
        if at in _SWIPE_SCROLL_TYPES and not (self.label_id or self.bbox):
            raise ValueError(
                f"label_id or bbox is required for action_type='{at}'. "
                "Ground every scroll action to a manifest container or an explicit visible region."
            )

        # wait_subject is required for wait actions.
        if at == "wait" and not (self.wait_subject or "").strip():
            raise ValueError(
                "wait_subject is required for action_type='wait'. "
                "Describe what we're waiting for (e.g., 'app to load', 'search results to appear')."
            )

        # capture is required for (and only valid for) store actions.
        if at == ActionType.STORE.value:
            if self.capture is None:
                raise ValueError(
                    "capture is required for action_type='store'. Provide name, subject, and value."
                )
        elif self.capture is not None:
            raise ValueError(f"capture is only valid for action_type='store', not '{at}'.")

        if at == ActionType.VALIDATE.value and not (self.validation_subject or "").strip():
            raise ValueError(
                "validation_subject is required for action_type='validate'. "
                "Provide the state or subject being asserted."
            )

        return self

    @model_validator(mode="after")
    def __normalize_conditionals(self) -> "ExecuteAction":
        action_type = (self.action_type or "").strip().lower()
        wait_subject = (self.wait_subject or "").strip() or None

        condition = (self.condition or "").strip() or None
        conditional_type = self.conditional_type
        is_conditional = self.is_conditional or bool(self.overlay_detected)

        # overlay_detected is a system signal; the guard text still belongs to the planner.
        if self.overlay_detected and not conditional_type:
            conditional_type = "blocker"

        # A conditional wait is "wait for X to appear"; the subject is the de-facto
        # guard. Mirror the overlay_detected default branch so the model is not
        # forced to repeat the same phrase twice.
        if (
            is_conditional
            and not condition
            and action_type == ActionType.WAIT.value
            and wait_subject
        ):
            condition = f"{wait_subject} is visible"
        if is_conditional and not conditional_type and action_type == ActionType.WAIT.value:
            conditional_type = "transient"

        # For all other conditional actions, require explicit condition text.
        if is_conditional and not condition:
            raise ValueError(
                "condition is required when is_conditional=True. "
                "Provide the guard condition text (e.g., 'Popup is visible', "
                "'Permission dialog is displayed', 'Loading spinner is active')."
            )

        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "conditional_type", conditional_type)
        object.__setattr__(self, "is_conditional", is_conditional)
        return self


class ExecuteUIArgs(GeminiCompletionFlags, GeminiDeltaTelemetry):
    """
    Schema for the execute_ui tool.
    """

    assistant_message: str = Field(
        "",
        description="Reasoning for the chosen UI action.",
    )
    action: Optional[ExecuteAction] = Field(
        default=None,
        description="Single low-level UI action for this turn.",
    )
    actions: List[ExecuteAction] = Field(
        default_factory=list,
        description="Legacy compatibility field. At most one action is accepted.",
    )
    memory_updates: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def __normalize_single_action(self) -> "ExecuteUIArgs":
        """
        Normalize the temporary legacy actions list into the singular action contract.
        """

        if self.action is not None and self.actions:
            raise ValueError("execute_ui accepts either action or actions, not both.")

        if len(self.actions) > 1:
            raise ValueError(
                "execute_ui accepts exactly one action; multi-action payloads are unsupported."
            )

        if self.action is None and self.actions:
            object.__setattr__(self, "action", self.actions[0])

        return self


class StoreMemoryArgs(BaseModel):
    """
    Schema for the store_memory tool.
    """

    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    assistant_message: str = Field(
        "",
        description="Explanation of why this memory is useful.",
    )


class RecallMemoryArgs(BaseModel):
    """
    Schema for the recall_memory tool.
    """

    key: str = Field(..., min_length=1)
    assistant_message: str = Field(
        "",
        description="Context for the memory lookup.",
    )


class AskUserArgs(GeminiCompletionFlags):
    """
    Schema for the ask_user tool.
    """

    question: str = Field(
        "",
        description="User-facing question to ask for clarification or input.",
    )
    context: str = Field(
        "",
        description="Optional context to help the user answer.",
    )
