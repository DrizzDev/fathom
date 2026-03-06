from __future__ import annotations

from collections import Counter
from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator


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
        if "```" in self.script:
            raise ValueError("Script must not contain markdown code fences.")

        lines: List[str] = [line.strip() for line in self.script.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Script has no meaningful lines.")

        brace_balance = 0
        for line in lines:
            if line.startswith("IF ") and line.endswith("{"):
                brace_balance += 1
                continue
            if line == "}":
                brace_balance -= 1
                if brace_balance < 0:
                    raise ValueError("Script has unmatched closing brace.")

        if brace_balance != 0:
            raise ValueError("Script has unbalanced IF block braces.")

        last_non_structural = ""
        for line in reversed(lines):
            if line == "}":
                continue
            if line.startswith("IF ") and line.endswith("{"):
                continue
            last_non_structural = line
            break

        if not last_non_structural.lower().startswith("validate"):
            raise ValueError("Script must end with a goal validation line.")

        if self.allowed_action_lines:
            allowed_counts = Counter(
                line.strip().lower() for line in self.allowed_action_lines if line.strip()
            )
            seen_counts: Counter[str] = Counter()

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

            for line in lines:
                lowered = line.lower()
                if not any(lowered.startswith(prefix) for prefix in executable_prefixes):
                    continue

                seen_counts[lowered] += 1
                if seen_counts[lowered] > allowed_counts.get(lowered, 0):
                    raise ValueError(
                        f"Executable line not present in step data or repeated too many times: {line}"
                    )

        return self
