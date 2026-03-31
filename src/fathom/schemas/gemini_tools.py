from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from fathom.schemas.validators import enforce_validate_prefix

CoordSystem = Literal["normalized", "pixel"]
ConditionalType = Literal["blocker", "transient", "error", "optional"]
TargetType = Literal["stable", "positional", "dynamic"]
WaitPattern = Literal["ad", "splash", "load", "search", "generic"]
TargetElementType = Literal["button", "icon", "option", "link", "field", "text", "checkbox"]
ValidationPattern = Literal["blocker", "transient", "error", "generic"]


class EmitScriptConditionalBlockArgs(BaseModel):
    """
    Raw conditional block arguments for the emit_script tool.

    Performs light type coercion. Semantic invariants (e.g., non-empty
    conditions, required IDs) are enforced by EmitScriptArgs validators
    and ScriptExportStructuredPayload.enforce_policy in export.py.
    """

    condition: Optional[str] = None
    condition_type: Optional[ConditionalType] = Field(
        default=None,
        description=(
            "Classification of this condition: blocker (popup/permission/consent), "
            "transient (loading/splash), error (error message), or optional (nice-to-have check)."
        ),
    )
    action_ids: List[str] = Field(
        default_factory=list,
        description="Executable action IDs under this condition.",
    )

    @field_validator("condition", mode="before")
    @classmethod
    def _normalize_condition(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("action_ids", mode="before")
    @classmethod
    def _coerce_action_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        return [str(value)] if str(value).strip() else []


class EmitScriptArgs(BaseModel):
    """
    Raw emit_script tool arguments as produced by Gemini.

    Enforces structural invariants at parse time:
    - final_validation must start with "Validate"
    - action_validations values must start with "Validate"
    - No duplicate action IDs within remaining_action_ids or conditional blocks
    - No empty conditional blocks
    """

    conditional_blocks: List[EmitScriptConditionalBlockArgs] = Field(
        default_factory=list,
        description="Ordered IF blocks for condition-scoped actions using action IDs.",
    )
    remaining_action_ids: List[str] = Field(
        default_factory=list,
        description="Ordered executable action IDs outside IF blocks.",
    )
    action_validations: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of action_id -> intermediate validation after that action; values start with 'Validate'. "
            "Mid-flow checks only—not the terminal script line."
        ),
    )
    final_validation: str = Field(
        ...,
        min_length=10,
        description=(
            "Single terminal UI-state line after the last catalog action; MUST start with 'Validate'. "
            "State visible/displayed only—no tap/click/type/select/navigate/search phrasing."
        ),
    )

    @field_validator("final_validation", mode="before")
    @classmethod
    def _enforce_final_validation_format(cls, value: Any) -> str:
        if value is None:
            raise ValueError(
                "final_validation is required. Provide a terminal validation line "
                "starting with 'Validate' (e.g., 'Validate cart page is displayed.')."
            )
        return enforce_validate_prefix(str(value), "final_validation")

    @field_validator("remaining_action_ids", mode="before")
    @classmethod
    def _coerce_remaining_ids(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        return [str(value)] if str(value).strip() else []

    @field_validator("action_validations", mode="before")
    @classmethod
    def _coerce_and_validate_action_validations(cls, value: Any) -> Dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            return {}
        cleaned: Dict[str, str] = {}
        for key, line in value.items():
            aid = str(key).strip()
            text = str(line).strip()
            if not aid or not text:
                continue
            enforce_validate_prefix(text, f"action_validations['{aid}']")
            cleaned[aid] = text
        return cleaned

    @model_validator(mode="after")
    def _deduplicate_and_clean_blocks(self) -> "EmitScriptArgs":
        # Deduplicate remaining_action_ids (preserve order).
        seen: set[str] = set()
        deduped: list[str] = []
        for aid in self.remaining_action_ids:
            if aid not in seen:
                seen.add(aid)
                deduped.append(aid)
        self.remaining_action_ids = deduped

        # Deduplicate within conditional blocks; reject empty blocks.
        for i, block in enumerate(self.conditional_blocks):
            if not block.action_ids:
                raise ValueError(f"conditional_blocks[{i}] has no action_ids; remove empty blocks.")
            block_seen: set[str] = set()
            block_deduped: list[str] = []
            for aid in block.action_ids:
                if aid not in block_seen:
                    block_seen.add(aid)
                    block_deduped.append(aid)
            block.action_ids = block_deduped

        return self


class GeminiBBox(BaseModel):
    """
    Lightweight bbox schema used at the tool boundary.

    Downstream we map this into the core Bounds model in actions.py.
    """

    x: int = Field(0, description="Center X coordinate")
    y: int = Field(0, description="Center Y coordinate")
    width: int = Field(0, ge=0, description="Width (0 for center-point predictions)")
    height: int = Field(0, ge=0, description="Height (0 for center-point predictions)")
    coord_system: CoordSystem = Field(
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

    bbox: Optional[GeminiBBox] = None
    action_type: str = Field(
        "wait",
        description="Low-level action type; mapped to internal ActionType.",
    )
    target_name: Optional[str] = None
    element_name: Optional[str] = None
    text: Optional[str] = None
    text_to_type: Optional[str] = None
    wait_duration: Optional[float] = None
    validation_reason: Optional[str] = None

    condition: Optional[str] = None
    is_conditional: bool = False
    conditional_type: Optional[ConditionalType] = None
    overlay_detected: bool = False

    target_type: Optional[TargetType] = None
    script_target: Optional[str] = None

    # Structured export signals — authoritative; no heuristic fallback.
    export_target: Optional[str] = None
    scroll_target: Optional[str] = None
    wait_subject: Optional[str] = None
    wait_pattern: Optional[WaitPattern] = None
    is_app_launcher: bool = False
    target_is_generic: Optional[bool] = None
    target_element_type: Optional[TargetElementType] = None
    validation_subject: Optional[str] = None
    validation_pattern: Optional[ValidationPattern] = None

    rationale: Optional[str] = None
    is_valid: bool = True
    confidence: float = 0.5  # Conservative default; VLM should provide explicit confidence
    label_id: Optional[str] = None

    @field_validator("wait_duration", mode="before")
    @classmethod
    def _parse_wait_duration(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("label_id", mode="before")
    @classmethod
    def _to_str_label(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    @field_validator("export_target", mode="before")
    @classmethod
    def _reject_generic_export_target(cls, value: Any) -> Optional[str]:
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
    def _enforce_structured_signals(self) -> "ExecuteAction":
        at = (self.action_type or "").strip().lower()

        # scroll_target is required for swipe/scroll actions.
        if at in _SWIPE_SCROLL_TYPES and not (self.scroll_target or "").strip():
            raise ValueError(
                f"scroll_target is required for action_type='{at}'. "
                "Provide the element or section being scrolled to find."
            )

        # wait_subject is required for wait actions.
        if at == "wait" and not (self.wait_subject or "").strip():
            raise ValueError(
                "wait_subject is required for action_type='wait'. "
                "Describe what we're waiting for (e.g., 'app to load', 'search results to appear')."
            )

        # validation_subject is required for validate actions.
        if at == "validate" and not (self.validation_subject or "").strip():
            raise ValueError(
                "validation_subject is required for action_type='validate'. "
                "Describe what is being validated (e.g., 'login status', 'cart is empty')."
            )

        return self

    @model_validator(mode="after")
    def _normalize_conditionals(self) -> "ExecuteAction":
        condition = (self.condition or "").strip() or None
        conditional_type = self.conditional_type
        is_conditional = self.is_conditional or bool(self.overlay_detected)

        # overlay_detected is a system signal — provide sensible defaults.
        if self.overlay_detected and not condition:
            condition = "Overlay is visible"
        if self.overlay_detected and not conditional_type:
            conditional_type = "blocker"

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
    actions: List[ExecuteAction] = Field(
        default_factory=list,
        description="List of candidate low-level UI actions; first is executed.",
    )
    memory_updates: Optional[Dict[str, str]] = None


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
