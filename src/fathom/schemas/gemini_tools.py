from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

CoordSystem = Literal["normalized", "pixel"]
ConditionalType = Literal["blocker", "transient", "error", "optional"]
TargetType = Literal["stable", "positional", "dynamic"]


class EmitScriptConditionalBlockArgs(BaseModel):
    """
    Raw conditional block arguments for the emit_script tool.

    This model mirrors the Gemini tool JSON schema but stays permissive:
    - condition text is optional (empty strings are normalized to None).
    - action_ids are coerced to a list of strings.
    All semantic invariants (e.g., non-empty conditions, required IDs) are
    enforced later by ScriptExportStructuredPayload in export.py.
    """

    condition: Optional[str] = None
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

    This model is intentionally lax and only performs light type coercion so
    that downstream code can distinguish:
      - raw provider output that is structurally parseable
      - normalization/validation failures applied later
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
        description="Optional map of action_id -> intermediate validation line.",
    )
    final_validation: Optional[str] = Field(
        default=None,
        description="Final goal validation line, typically starting with 'Validate'.",
    )

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
    def _coerce_action_validations(cls, value: Any) -> Dict[str, str]:
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
            cleaned[aid] = text
        return cleaned


class GeminiBBox(BaseModel):
    """
    Lightweight bbox schema used at the tool boundary.

    Downstream we map this into the core Bounds model in actions.py.
    """

    x: int = Field(0, description="Top-left X coordinate")
    y: int = Field(0, description="Top-left Y coordinate")
    width: int = Field(..., gt=0, description="Bounding box width")
    height: int = Field(..., gt=0, description="Bounding box height")
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

    @model_validator(mode="after")
    def _fill_missing_sub_goal(self) -> "VerifyGoalArgs":
        if self.sub_goal_completed is None:
            object.__setattr__(self, "sub_goal_completed", self.goal_completed)
        return self


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

    @model_validator(mode="after")
    def _default_flags_from_condition(self) -> "ValidateStateArgs":
        if self.goal_completed is None:
            object.__setattr__(self, "goal_completed", False)
        if self.sub_goal_completed is None:
            completed = bool(self.condition_met)
            object.__setattr__(self, "sub_goal_completed", completed)
        return self


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

    rationale: Optional[str] = None
    is_valid: bool = True
    confidence: float = 1.0
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

    @model_validator(mode="after")
    def _normalize_conditionals(self) -> "ExecuteAction":
        condition = (self.condition or "").strip() or None
        conditional_type = self.conditional_type
        is_conditional = self.is_conditional or bool(self.overlay_detected)

        if self.overlay_detected and not condition:
            condition = "Overlay is visible"
        if self.overlay_detected and not conditional_type:
            conditional_type = "blocker"

        if is_conditional and not condition:
            default_map = {
                "blocker": "Blocker prompt is visible",
                "transient": "Transient screen is visible",
                "error": "Error message is displayed",
                "optional": "Optional UI state is visible",
            }
            condition = default_map.get(conditional_type or "", "Conditional UI state is visible")

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

    @model_validator(mode="after")
    def _default_completion_flags(self) -> "AskUserArgs":
        if self.goal_completed is None:
            object.__setattr__(self, "goal_completed", False)
        if self.sub_goal_completed is None:
            object.__setattr__(self, "sub_goal_completed", False)
        return self
