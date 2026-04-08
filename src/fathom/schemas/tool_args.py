from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from fathom.constants import ActionType
from fathom.schemas.actions import GENERIC_TARGET_PLACEHOLDERS
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

    x: int = Field(
        0,
        description="X coordinate (center for VLM predictions, top-left for label-snapped bounds)",
    )
    y: int = Field(
        0,
        description="Y coordinate (center for VLM predictions, top-left for label-snapped bounds)",
    )
    width: int = Field(
        0,
        ge=0,
        description="Width (0 for VLM center-point predictions, >0 for label-snapped bounds)",
    )
    height: int = Field(
        0,
        ge=0,
        description="Height (0 for VLM center-point predictions, >0 for label-snapped bounds)",
    )
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
        default=False,
        description="Whether the overall user goal is complete. Only set by verify_goal.",
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


_SWIPE_SCROLL_TYPES: frozenset[str] = frozenset(
    {
        ActionType.SWIPE_UP,
        ActionType.SWIPE_DOWN,
        ActionType.SWIPE_LEFT,
        ActionType.SWIPE_RIGHT,
        ActionType.SCROLL,
    }
)
_PHYSICAL_BBOX_TYPES: frozenset[str] = frozenset(
    {
        ActionType.TAP,
        ActionType.TYPE,
        ActionType.LONG_PRESS,
        ActionType.SWIPE_UP,
        ActionType.SWIPE_DOWN,
        ActionType.SWIPE_LEFT,
        ActionType.SWIPE_RIGHT,
    }
)
# back/home need no target; validate uses validation_subject; swipe_* uses
# scroll_target; wait uses wait_subject — only direct-touch actions need a
# user-facing target name.
_EXPORT_TARGET_REQUIRED_TYPES: frozenset[str] = frozenset(
    {
        ActionType.TAP,
        ActionType.TYPE,
        ActionType.LONG_PRESS,
    }
)
# Single source of truth lives in fathom.schemas.actions; alias here for the
# field validator that runs before the model validator.
_GENERIC_EXPORT_TARGETS = GENERIC_TARGET_PLACEHOLDERS

# Default guard text per conditional class. Used when the LLM sets
# is_conditional=True (or implies it via conditional_type / overlay_detected)
# without providing explicit condition text. Mirrors the prompt's contract:
# "if condition is omitted, conditional_type is used for default guard text".
_DEFAULT_CONDITION_TEXT: Dict[str, str] = {
    "blocker": "Blocking overlay is visible",
    "transient": "Loading state is active",
    "error": "Error message is visible",
    "optional": "Optional element is visible",
}


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
    def _normalize_target_fields(self) -> "ExecuteAction":
        """Resolve canonical target_name and derive export_target.

        Accepts ``element_name`` and explicit ``export_target`` as
        deprecated aliases. ``export_target`` is derived from
        ``script_target`` when ``target_type`` is positional/dynamic, else
        from ``target_name``. Generic placeholders are skipped.
        """

        def first_clean(*candidates: Optional[str]) -> Optional[str]:
            for candidate in candidates:
                if candidate is None:
                    continue
                text = str(candidate).strip()
                if not text:
                    continue
                if text.lower() in _GENERIC_EXPORT_TARGETS:
                    continue
                return text
            return None

        # 1) Resolve the canonical on-screen element name.
        canonical_target_name = first_clean(
            self.target_name,
            self.element_name,
            self.script_target,
        )
        if canonical_target_name and canonical_target_name != self.target_name:
            object.__setattr__(self, "target_name", canonical_target_name)

        # 2) Derive export_target. For positional/dynamic targets the
        #    abstracted script_target wins; otherwise fall back to the
        #    canonical target_name. An explicit export_target from the
        #    caller still takes priority for back-compat.
        if not (self.export_target or "").strip():
            if self.target_type in ("positional", "dynamic"):
                derived_export = first_clean(
                    self.script_target,
                    canonical_target_name,
                )
            else:
                derived_export = first_clean(
                    canonical_target_name,
                    self.script_target,
                )
            if derived_export:
                object.__setattr__(self, "export_target", derived_export)

        return self

    @model_validator(mode="after")
    def _enforce_bbox_for_physical_actions(self) -> "ExecuteAction":
        at = (self.action_type or "").strip().lower()
        if at not in _PHYSICAL_BBOX_TYPES:
            return self

        # GROUNDING FIRST: when label_id is set, the manifest snapper will
        # provide exact label-aligned bounds downstream and the LLM-emitted
        # bbox is just an approximation. Demanding a bbox here would
        # contradict the prompt's "ALWAYS prefer label_id" instruction.
        if (self.label_id or "").strip():
            return self

        bbox = self.bbox
        bbox_missing = bbox is None or (
            bbox.x == 0 and bbox.y == 0 and bbox.width == 0 and bbox.height == 0
        )
        if bbox_missing:
            raise ValueError(
                f"bbox with non-zero coordinates is required for action_type='{at}' "
                "when label_id is not provided. Either supply label_id from the "
                "Element Manifest, or provide x,y at the CENTER of the target "
                "element using normalized values (0-1000)."
            )
        return self

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

        # validation_subject is required for validate actions. The
        # downstream Action._enforce_validation_subject validator (in
        # fathom.schemas.actions) catches first-person/narrative prose
        # using the canonical VALIDATION_SUBJECT_BAD_PREFIXES list — do
        # not duplicate that check here.
        if at == "validate" and not (self.validation_subject or "").strip():
            raise ValueError(
                "validation_subject is required for action_type='validate'. "
                "Describe what is being validated (e.g., 'login status', 'cart is empty')."
            )

        # export_target is required for actions that render to an exported
        # script line (tap, type, long_press). Not required for back/home
        # (device buttons, no target), validate (uses validation_subject),
        # swipe_* (uses scroll_target), or wait (uses wait_subject).
        # GROUNDING FIRST: if label_id is set, the manifest provides the
        # canonical element name at bind time — the target_name field
        # is not the authoritative source in that case, so we skip the
        # check to match the prompt's "prefer label_id" instruction.
        if (
            at in _EXPORT_TARGET_REQUIRED_TYPES
            and not (self.export_target or "").strip()
            and not (self.label_id or "").strip()
        ):
            raise ValueError(
                f"export_target is required for action_type='{at}' "
                "when label_id is not provided. Either supply label_id from "
                "the Element Manifest, or provide a canonical phrase for the "
                "exported test script (e.g., 'Search box', 'the first search "
                "result')."
            )

        return self

    @model_validator(mode="after")
    def _normalize_conditionals(self) -> "ExecuteAction":
        """Resolve the conditional cluster from any LLM-provided signal.

        ``is_conditional`` is implied by ANY conditional signal
        (``overlay_detected``, a non-empty ``condition``, or a set
        ``conditional_type``). When the LLM sets the flag without
        condition text, the text is derived from ``conditional_type``
        (defaulting to ``blocker`` for ``overlay_detected``). Only fail
        when the action is conditional but carries no signal we can
        derive a guard string from.
        """

        condition = (self.condition or "").strip() or None
        conditional_type = self.conditional_type

        # overlay_detected implies blocker class.
        if self.overlay_detected and not conditional_type:
            conditional_type = "blocker"

        # Any of these signals means the action is conditional.
        is_conditional = (
            self.is_conditional
            or bool(self.overlay_detected)
            or conditional_type is not None
            or condition is not None
        )

        # Derive default condition text from conditional_type when missing.
        # Matches the prompt's "conditional_type is used for default guard text".
        if is_conditional and not condition and conditional_type:
            condition = _DEFAULT_CONDITION_TEXT.get(conditional_type)

        # Only fail when we genuinely have nothing to render as a guard.
        if is_conditional and not condition:
            raise ValueError(
                "is_conditional=True requires either condition text or "
                "conditional_type so a default guard string can be derived."
            )

        object.__setattr__(self, "condition", condition)
        object.__setattr__(self, "conditional_type", conditional_type)
        object.__setattr__(self, "is_conditional", is_conditional)
        return self


_TARGET_INHERIT_FIELDS: tuple[str, ...] = (
    "target_name",
    "element_name",
    "script_target",
    "export_target",
    "target_type",
    "target_element_type",
    "target_is_generic",
)


def _inherit_target_from_prior_action(action: Dict[str, Any], prior: Dict[str, Any]) -> None:
    """Copy target fields from a prior action when the current one elides them.

    Mutates ``action`` in place. Used when Gemini emits the common
    tap-then-type pattern: the tap fully describes the target field, the
    type re-uses the same coordinates and label but skips repeating the
    target name. Without this, the second action fails the per-action
    target_name requirement even though it is unambiguously bound to the
    prior action's target.
    """

    has_target = any(
        (action.get(key) or "")
        for key in ("target_name", "element_name", "script_target", "export_target")
    )
    if has_target:
        return

    same_label = (
        action.get("label_id")
        and prior.get("label_id")
        and str(action["label_id"]) == str(prior["label_id"])
    )

    same_bbox = False
    a_bbox = action.get("bbox") if isinstance(action.get("bbox"), dict) else None
    p_bbox = prior.get("bbox") if isinstance(prior.get("bbox"), dict) else None
    if a_bbox and p_bbox:
        ax, ay = a_bbox.get("x"), a_bbox.get("y")
        px, py = p_bbox.get("x"), p_bbox.get("y")
        if all(v is not None for v in (ax, ay, px, py)):
            same_bbox = abs(int(ax) - int(px)) <= 5 and abs(int(ay) - int(py)) <= 5

    if not (same_label or same_bbox):
        return

    for key in _TARGET_INHERIT_FIELDS:
        if not action.get(key) and prior.get(key) is not None:
            action[key] = prior[key]


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

    @field_validator("actions", mode="before")
    @classmethod
    def _propagate_targets(cls, value: Any) -> Any:
        """Inherit target fields between consecutive actions on the same element.

        Runs BEFORE per-action validation so a type-after-tap action that
        omits target_name still has it filled in from the prior tap when
        they share label_id (or a near-identical bbox center).
        """

        if not isinstance(value, list):
            return value

        normalized: list[Any] = []
        prior_dict: Optional[Dict[str, Any]] = None
        for entry in value:
            if isinstance(entry, dict):
                _inherit_target_from_prior_action(entry, prior_dict or {})
                prior_dict = entry
            else:
                # Already-constructed ExecuteAction or unknown type:
                # nothing to mutate. Skip but still update prior so the
                # next dict can read its target fields if needed.
                prior_dict = None
            normalized.append(entry)
        return normalized


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
