from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ConditionalBlockPayload(BaseModel):
    """
    Structured conditional block for script generation.
    """

    condition: str = Field(min_length=1, description="IF block condition text.")
    action_ids: List[str] = Field(
        default_factory=list,
        description="Executable action IDs under this condition.",
    )


class ScriptExportStructuredPayload(BaseModel):
    """
    Structured Gemini payload that is rendered into script text.
    """

    conditional_blocks: List[ConditionalBlockPayload] = Field(
        default_factory=list, description="Ordered IF blocks."
    )
    remaining_action_ids: List[str] = Field(
        default_factory=list, description="Ordered executable action IDs outside IF blocks."
    )
    final_validation: str = Field(
        min_length=1, description="Final goal validation line starting with 'Validate'."
    )
    action_validations: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of action_id -> intermediate validation line. "
            "Each value must start with 'Validate' and is emitted after that action."
        ),
    )

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

    @model_validator(mode="after")
    def __validate_against_allowed_actions(self) -> "ScriptExportStructuredPayload":
        if not self.final_validation.strip().lower().startswith("validate"):
            raise ValueError("final_validation must start with 'Validate'.")

        cleaned_action_validations: Dict[str, str] = {}
        for action_id, validation_line in self.action_validations.items():
            aid = str(action_id).strip()
            line = str(validation_line).strip()
            if not aid or not line:
                continue
            if not line.lower().startswith("validate"):
                raise ValueError(f"action_validations[{aid}] must start with 'Validate'.")
            cleaned_action_validations[aid] = line
        self.action_validations = cleaned_action_validations

        non_empty_blocks = [
            block
            for block in self.conditional_blocks
            if any(action_id.strip() for action_id in block.action_ids)
        ]
        if len(non_empty_blocks) != len(self.conditional_blocks):
            self.conditional_blocks = non_empty_blocks

        if self.require_if_block and not self.conditional_blocks:
            raise ValueError("At least one conditional block is required for this intent.")

        ordered_action_ids: List[str] = []
        for block in self.conditional_blocks:
            ordered_action_ids.extend(
                action_id.strip() for action_id in block.action_ids if action_id.strip()
            )
        ordered_action_ids.extend(
            action_id.strip() for action_id in self.remaining_action_ids if action_id.strip()
        )

        if not ordered_action_ids:
            raise ValueError("No executable action IDs were provided.")

        if self.required_open_app_id:
            if self.required_open_app_id not in ordered_action_ids:
                raise ValueError(
                    f"Required OPEN_APP action ID is missing: {self.required_open_app_id}"
                )
            ordered_action_ids = [self.required_open_app_id] + [
                action_id
                for action_id in ordered_action_ids
                if action_id != self.required_open_app_id
            ]
            if ordered_action_ids[0] != self.required_open_app_id:
                raise ValueError(
                    f"First executable action ID must be exactly: {self.required_open_app_id}"
                )

        if self.required_action_ids and Counter(ordered_action_ids) != Counter(
            self.required_action_ids
        ):
            raise ValueError(
                "Executable action IDs must match step data exactly (no missing, extra, or duplicated IDs)."
            )

        for action_id in ordered_action_ids:
            if action_id not in self.action_catalog:
                raise ValueError(f"Unknown action ID referenced: {action_id}")

        for action_id in self.action_validations:
            if action_id not in self.action_catalog:
                raise ValueError(f"Unknown action ID in action_validations: {action_id}")
            if self.action_catalog[action_id].strip().lower().startswith("open_app "):
                raise ValueError("action_validations cannot target OPEN_APP actions.")

        # Enforce validation distribution when multiple validations are expected
        if self.expected_validation_count > 1:
            total_validations = len(self.action_validations) + 1  # +1 for final_validation
            if total_validations < self.expected_validation_count:
                raise ValueError(
                    f"Intent requires {self.expected_validation_count} validations, "
                    f"but only {total_validations} were provided. "
                    f"Expected at least {self.expected_validation_count - 1} intermediate validations in action_validations "
                    f"(found {len(self.action_validations)})."
                )

        return self

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
                lines.append(f"IF {block.condition.strip()} {{")
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

                if current.startswith("IF "):
                    if "{" not in current:
                        continue
                    current = current.split("{", 1)[1].strip()

                current = current.strip("{} ").strip()
                if current:
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
            has_if_block = any(line.lower().startswith("if ") and "{" in line for line in lines)
            if not has_if_block:
                raise ValueError("Script must contain at least one IF conditional block.")

        statements = __extract_action_statements(raw_lines=lines)
        last_non_structural = statements[-1] if statements else ""

        if not last_non_structural.lower().startswith("validate"):
            raise ValueError("Script must end with a goal validation line.")

        executable_prefixes = (
            "open_app ",
            "tap ",
            "type ",
            "scroll ",
            "swipe ",
            "wait ",
            "press ",
            "long press ",
        )
        executable_statements = [
            statement.lower()
            for statement in statements
            if statement.lower().startswith(executable_prefixes)
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
                if not any(lowered.startswith(prefix) for prefix in executable_prefixes):
                    continue

                seen_counts[lowered] += 1
                if seen_counts[lowered] > allowed_counts.get(lowered, 0):
                    raise ValueError(
                        f"Executable line not present in step data or repeated too many times: {statement}"
                    )

        return self
