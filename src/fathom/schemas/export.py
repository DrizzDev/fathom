from __future__ import annotations

from collections import Counter
from logging import getLogger
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from fathom.constants import EXECUTABLE_ACTION_PREFIXES, VALIDATE_PREFIX

logger = getLogger(__name__)


class ConditionalBlockPayload(BaseModel):
    """
    Structured conditional block for script generation.
    """

    condition: str = Field(
        min_length=1,
        description="IF block condition text. Rendered as 'IF <condition>' with '{' on the following line.",
    )
    condition_type: Optional[Literal["blocker", "transient", "error", "optional"]] = Field(
        default=None,
        description="Classification of this condition: blocker, transient, error, or optional.",
    )
    action_ids: List[str] = Field(
        default_factory=list,
        description="Executable action IDs under this condition.",
    )


class ScriptExportStructuredPayloadShape(BaseModel):
    """
    Shape-only structured Gemini payload (schema validation only).

    Export policy/rule enforcement (e.g., OPEN_APP requirements, action coverage,
    canonical ordering) is handled separately by `ScriptExportStructuredPayload.enforce_policy`.
    """

    conditional_blocks: List[ConditionalBlockPayload] = Field(
        default_factory=list, description="Ordered IF blocks."
    )
    remaining_action_ids: List[str] = Field(
        default_factory=list, description="Ordered executable action IDs outside IF blocks."
    )
    final_validation: str = Field(
        min_length=1,
        description=(
            "Terminal UI-state validation after the last catalog action; must start with 'Validate'. "
            "Single concise visible/displayed assertion—no imperative tap/click/type steps."
        ),
    )
    action_validations: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of action_id -> intermediate validation emitted after that action. "
            "Each value must start with 'Validate'. Use for mid-flow state checks; not for the terminal line."
        ),
    )


class ScriptExportStructuredPayload(ScriptExportStructuredPayloadShape):
    """
    Structured payload bound to an export context (catalog + requirements).

    This model is used for rendering (`to_script`). It does not embed export policy
    enforcement in Pydantic validators; call `enforce_policy(...)` explicitly.
    """

    action_catalog: Dict[str, str] = Field(
        default_factory=dict,
        description="Action catalog mapping action IDs to executable action lines.",
        exclude=True,
        repr=False,
    )
    required_action_ids: List[str] = Field(
        default_factory=list,
        description="All required executable action IDs from step data.",
        exclude=True,
        repr=False,
    )
    required_open_app_id: Optional[str] = Field(
        default=None,
        description="Required OPEN_APP action ID when package is known.",
        exclude=True,
        repr=False,
    )
    require_if_block: bool = Field(
        default=False,
        description="Require at least one IF block for conditional intents.",
        exclude=True,
        repr=False,
    )
    expected_validation_count: int = Field(
        default=1,
        description="Number of validation subjects extracted from user intent. Used to enforce validation distribution.",
        exclude=True,
        repr=False,
    )

    @classmethod
    def enforce_policy(
        cls,
        *,
        shape: ScriptExportStructuredPayloadShape,
        action_catalog: Dict[str, str],
        required_action_ids: List[str],
        required_open_app_id: Optional[str],
        require_if_block: bool,
        expected_validation_count: int,
    ) -> "ScriptExportStructuredPayload":
        """
        Apply export policy/rule enforcement to an already shape-validated payload.

        Raises:
            ValueError: when policy invariants are violated.
        """

        payload = cls.model_validate(
            {
                **shape.model_dump(),
                "action_catalog": action_catalog,
                "required_action_ids": required_action_ids,
                "required_open_app_id": required_open_app_id,
                "require_if_block": require_if_block,
                "expected_validation_count": expected_validation_count,
            }
        )

        if not payload.final_validation.strip().lower().startswith(VALIDATE_PREFIX):
            raise ValueError("final_validation must start with 'Validate'.")

        cleaned_action_validations: Dict[str, str] = {}
        for action_id, validation_line in payload.action_validations.items():
            aid = str(action_id).strip()
            line = str(validation_line).strip()
            if not aid or not line:
                continue
            if not line.lower().startswith(VALIDATE_PREFIX):
                raise ValueError(f"action_validations[{aid}] must start with 'Validate'.")
            cleaned_action_validations[aid] = line
        payload.action_validations = cleaned_action_validations

        non_empty_blocks = [
            block
            for block in payload.conditional_blocks
            if any(action_id.strip() for action_id in block.action_ids)
        ]
        if len(non_empty_blocks) != len(payload.conditional_blocks):
            payload.conditional_blocks = non_empty_blocks

        if payload.require_if_block and not payload.conditional_blocks:
            raise ValueError("At least one conditional block is required for this intent.")

        ordered_action_ids: List[str] = []
        for block in payload.conditional_blocks:
            # Normalize condition text and enforce non-empty after stripping.
            condition_text = block.condition.strip()
            if not condition_text:
                raise ValueError("Conditional block condition must not be empty.")

            ordered_action_ids.extend(
                action_id.strip() for action_id in block.action_ids if action_id.strip()
            )
        ordered_action_ids.extend(
            action_id.strip() for action_id in payload.remaining_action_ids if action_id.strip()
        )

        if not ordered_action_ids:
            raise ValueError("No executable action IDs were provided.")

        if payload.required_open_app_id:
            if payload.required_open_app_id not in ordered_action_ids:
                # Auto-prepend the OPEN_APP action instead of hard-failing.
                # The LLM sometimes omits it because it considers it implicit.
                logger.warning(
                    "LLM omitted required OPEN_APP action %s; auto-prepending.",
                    payload.required_open_app_id,
                )
                ordered_action_ids.insert(0, payload.required_open_app_id)
            else:
                # Move it to position 0 if present but not first.
                ordered_action_ids = [payload.required_open_app_id] + [
                    action_id
                    for action_id in ordered_action_ids
                    if action_id != payload.required_open_app_id
                ]

        # Enforce that action IDs inside each conditional block respect the canonical
        # execution order when we have required_action_ids available. This keeps
        # conditionals structurally aligned with the underlying trace while leaving
        # Gemini free to choose which subset to include.
        if payload.required_action_ids:
            rank = {
                action_id.strip(): index
                for index, action_id in enumerate(payload.required_action_ids)
                if action_id.strip()
            }
            for block in payload.conditional_blocks:
                indices = [rank.get(aid.strip(), -1) for aid in block.action_ids if aid.strip()]
                filtered = [index for index in indices if index >= 0]
                if not filtered:
                    continue
                if any(b < a for a, b in zip(filtered, filtered[1:], strict=False)):
                    raise ValueError(
                        "Conditional block action_ids must follow the canonical step order."
                    )

        if payload.required_action_ids and Counter(ordered_action_ids) != Counter(
            payload.required_action_ids
        ):
            raise ValueError(
                "Executable action IDs must match step data exactly (no missing, extra, or duplicated IDs)."
            )

        for action_id in ordered_action_ids:
            if action_id not in payload.action_catalog:
                raise ValueError(f"Unknown action ID referenced: {action_id}")

        for action_id in payload.action_validations:
            if action_id not in payload.action_catalog:
                raise ValueError(f"Unknown action ID in action_validations: {action_id}")
            if payload.action_catalog[action_id].strip().lower().startswith("open_app "):
                raise ValueError("action_validations cannot target OPEN_APP actions.")

        # Reject degenerate duplicated conditional blocks that use the same condition
        # text but whose action ID sets are strict subsets of one another. This keeps
        # IF structure meaningful without inspecting the semantic content of conditions.
        if payload.conditional_blocks:
            normalized_blocks: List[tuple[str, set[str]]] = []
            for block in payload.conditional_blocks:
                condition_text = block.condition.strip().lower()
                id_set = {aid.strip() for aid in block.action_ids if aid.strip()}
                normalized_blocks.append((condition_text, id_set))

            for i in range(len(normalized_blocks)):
                cond_i, ids_i = normalized_blocks[i]
                if not ids_i:
                    continue
                for j in range(i + 1, len(normalized_blocks)):
                    cond_j, ids_j = normalized_blocks[j]
                    if cond_i != cond_j:
                        continue
                    if ids_i and ids_j and (ids_i < ids_j or ids_j < ids_i):
                        raise ValueError(
                            "Degenerate duplicate conditional blocks detected for the same condition."
                        )

        # Log when validation distribution is sparse but do not reject — the LLM
        # may legitimately cover multiple subjects in a single validation statement.
        if payload.expected_validation_count > 1:
            total_validations = len(payload.action_validations) + 1  # +1 for final_validation
            if total_validations < 2:
                logger.warning(
                    "Intent has %d validation subjects but only %d validation statement(s) "
                    "were provided (%d intermediate + 1 final). The final_validation may "
                    "cover multiple subjects.",
                    payload.expected_validation_count,
                    total_validations,
                    len(payload.action_validations),
                )

        return payload

    def to_script(self) -> str:
        """
        Render structured payload into canonical script text.
        """

        lines: List[str] = []
        required_open_app_id = (
            self.required_open_app_id.strip() if self.required_open_app_id else None
        )
        if required_open_app_id:
            open_line = self.action_catalog.get(required_open_app_id, "").strip()
            if open_line:
                lines.append(open_line)

        # Build a deterministic chronological order of selected action IDs so IF blocks
        # are emitted where their first action occurs in execution order.
        selected_ids: List[str] = []
        for block in self.conditional_blocks:
            selected_ids.extend(
                action_id.strip() for action_id in block.action_ids if action_id.strip()
            )
        selected_ids.extend(
            action_id.strip() for action_id in self.remaining_action_ids if action_id.strip()
        )

        if self.required_action_ids:
            canonical_order = [
                action_id.strip() for action_id in self.required_action_ids if action_id.strip()
            ]
        else:
            canonical_order = list(self.action_catalog.keys())

        rank = {action_id: index for index, action_id in enumerate(canonical_order)}
        ordered_selected_ids = sorted(
            dict.fromkeys(selected_ids),
            key=lambda action_id: rank.get(action_id, len(rank)),
        )

        block_by_action_id: Dict[str, int] = {}
        for block_index, block in enumerate(self.conditional_blocks):
            for action_id in block.action_ids:
                action_id = action_id.strip()
                if action_id:
                    block_by_action_id[action_id] = block_index

        def __append_action_validation(action_id: str, *, indent: str = "") -> None:
            validation_line = self.action_validations.get(action_id, "").strip()
            if validation_line:
                lines.append(f"{indent}{validation_line}")

        emitted_block_indices: set[int] = set()
        emitted_non_block_action_ids: set[str] = set()
        for action_id in ordered_selected_ids:
            if required_open_app_id and action_id == required_open_app_id:
                continue

            selected_block_index = block_by_action_id.get(action_id)
            if selected_block_index is not None:
                if selected_block_index in emitted_block_indices:
                    continue
                block = self.conditional_blocks[selected_block_index]
                lines.append(f"IF {block.condition.strip()}")
                lines.append("{")
                for block_action_id in block.action_ids:
                    block_action_id = block_action_id.strip()
                    if not block_action_id:
                        continue
                    if required_open_app_id and block_action_id == required_open_app_id:
                        continue
                    action_line = self.action_catalog.get(block_action_id, "").strip()
                    if action_line:
                        lines.append(f"    {action_line}")
                        __append_action_validation(block_action_id, indent="    ")
                lines.append("}")
                emitted_block_indices.add(selected_block_index)
                continue

            if action_id in emitted_non_block_action_ids:
                continue
            action_line = self.action_catalog.get(action_id, "").strip()
            if action_line:
                lines.append(action_line)
                __append_action_validation(action_id)
                emitted_non_block_action_ids.add(action_id)

        lines.append(self.final_validation.strip())
        return "\n".join(lines).strip() + "\n"


class ScriptExportPayload(BaseModel):
    """
    Validated payload for exported script text produced by Gemini.
    """

    script: str = Field(min_length=1, description="Final script text content.")
    allowed_action_lines: List[str] = Field(
        default_factory=list,
        description="Allowed executable action lines derived from step data.",
        exclude=True,
        repr=False,
    )
    required_open_app: Optional[str] = Field(
        default=None,
        description="Required first executable OPEN_APP line when package is known.",
        exclude=True,
        repr=False,
    )
    require_if_block: bool = Field(
        default=False,
        description="Require at least one IF conditional block in the script.",
        exclude=True,
        repr=False,
    )

    @field_validator("script")
    @classmethod
    def __normalize_script(cls, value: str) -> str:
        lines = [line.rstrip() for line in str(value).replace("\r\n", "\n").split("\n")]
        while lines and not lines[-1].strip():
            lines.pop()
        normalized = "\n".join(lines).strip()
        if not normalized:
            raise ValueError("Script is empty.")
        return normalized + "\n"

    @model_validator(mode="after")
    def __validate_structure(self) -> "ScriptExportPayload":
        def __extract_action_statements(raw_lines: List[str]) -> List[str]:
            statements: List[str] = []
            for raw in raw_lines:
                current = raw.strip()
                if not current:
                    continue
                if current == "{" or current == "}":
                    continue
                lower = current.lower()
                if lower.startswith("if "):
                    if "{" not in current:
                        continue
                    tail = current.split("{", 1)[1].strip().rstrip("}").strip()
                    if tail:
                        statements.append(tail)
                    continue
                statements.append(current)
            return statements

        if "```" in self.script:
            raise ValueError("Script must not contain markdown code fences.")

        lines: List[str] = [line.strip() for line in self.script.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Script has no meaningful lines.")

        brace_balance = 0
        for line in lines:
            for character in line:
                if character == "{":
                    brace_balance += 1
                elif character == "}":
                    brace_balance -= 1
                    if brace_balance < 0:
                        raise ValueError("Script has unmatched closing brace.")

        if brace_balance != 0:
            raise ValueError("Script has unbalanced IF block braces.")

        if self.require_if_block:

            def __script_has_if_block(line_list: List[str]) -> bool:
                n = len(line_list)
                for idx, line in enumerate(line_list):
                    lower = line.lower()
                    if not lower.startswith("if "):
                        continue
                    if "{" in line:
                        return True
                    if idx + 1 < n and line_list[idx + 1] == "{":
                        return True
                return False

            if not __script_has_if_block(lines):
                raise ValueError("Script must contain at least one IF conditional block.")

            # Additionally require that at least one IF block guards a non-trivial body with
            # more than one non-structural line (e.g., an action and/or validation), to avoid
            # accepting degenerate empty or single-line shells.
            max_body_statements = 0
            in_if_block = False
            current_body_count = 0
            i = 0
            while i < len(lines):
                line = lines[i]
                lower = line.lower()
                if lower.startswith("if "):
                    if in_if_block:
                        max_body_statements = max(max_body_statements, current_body_count)
                    in_if_block = True
                    current_body_count = 0
                    if "{" in line:
                        tail = line.split("{", 1)[1].strip().rstrip("}").strip()
                        if tail:
                            current_body_count = 1
                        i += 1
                        continue
                    if i + 1 < len(lines) and lines[i + 1] == "{":
                        i += 2
                        continue
                    i += 1
                    continue
                if line == "}":
                    if in_if_block:
                        max_body_statements = max(max_body_statements, current_body_count)
                        in_if_block = False
                        current_body_count = 0
                    i += 1
                    continue
                if in_if_block:
                    current_body_count += 1
                i += 1

            if in_if_block:
                max_body_statements = max(max_body_statements, current_body_count)

            if max_body_statements <= 1:
                raise ValueError(
                    "Script must contain at least one non-trivial IF block with multiple statements."
                )

        statements = __extract_action_statements(raw_lines=lines)
        last_non_structural = statements[-1] if statements else ""

        if not last_non_structural.lower().startswith(VALIDATE_PREFIX):
            raise ValueError("Script must end with a goal validation line.")

        executable_statements = [
            statement.lower()
            for statement in statements
            if statement.lower().startswith(EXECUTABLE_ACTION_PREFIXES)
        ]

        required_open_app = (self.required_open_app or "").strip().lower()
        if required_open_app:
            if not executable_statements:
                raise ValueError("Script has no executable actions.")
            if executable_statements[0] != required_open_app:
                raise ValueError(f"First executable line must be exactly: {self.required_open_app}")

        if self.allowed_action_lines:
            allowed_counts = Counter(
                line.strip().lower() for line in self.allowed_action_lines if line.strip()
            )
            seen_counts: Counter[str] = Counter()

            for statement in statements:
                lowered = statement.lower()
                if not any(lowered.startswith(prefix) for prefix in EXECUTABLE_ACTION_PREFIXES):
                    continue

                seen_counts[lowered] += 1
                if seen_counts[lowered] > allowed_counts.get(lowered, 0):
                    raise ValueError(
                        f"Executable line not present in step data or repeated too many times: {statement}"
                    )

        return self
