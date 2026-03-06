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

    @model_validator(mode="after")
    def __validate_against_allowed_actions(self) -> "ScriptExportStructuredPayload":
        if not self.final_validation.strip().lower().startswith("validate"):
            raise ValueError("final_validation must start with 'Validate'.")

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

        for block in self.conditional_blocks:
            lines.append(f"IF {block.condition.strip()} {{")
            for action_id in block.action_ids:
                if required_open_app_id and action_id.strip() == required_open_app_id:
                    continue
                action_line = self.action_catalog.get(action_id.strip(), "").strip()
                if action_line:
                    lines.append(f"    {action_line}")
            lines.append("}")

        for action_id in self.remaining_action_ids:
            if required_open_app_id and action_id.strip() == required_open_app_id:
                continue
            action_line = self.action_catalog.get(action_id.strip(), "").strip()
            if action_line:
                lines.append(action_line)

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
