"""Schemas for LLM-backed intent decomposition."""

from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fathom.schemas.subgoal import ExecutionContract


class DecomposedTask(BaseModel):
    """
    Decomposed task with an observable terminal criterion.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = Field(min_length=1, description="Imperative task description.")
    criterion: str = Field(
        min_length=1,
        description="Observable screen state criterion satisfied when the task is complete.",
    )
    execution_contract: ExecutionContract = Field(
        default_factory=ExecutionContract,
        description="Structured execution constraints preserved by strict mode.",
    )


class DecompositionSchema(BaseModel):
    """
    Schema for intent decomposition output.
    """

    confidence: Optional[float] = 0.9
    sub_goals: List[Union[str, DecomposedTask]]

    @field_validator("sub_goals", mode="before")
    @classmethod
    def __coerce_sub_goals(cls, value: Any) -> List[Union[str, DecomposedTask]]:
        """
        Normalize entries to strings or DecomposedTask objects.
        """

        if not isinstance(value, list):
            raise ValueError("sub_goals must be a list")

        if not value:
            raise ValueError("sub_goals must not be empty")

        if len(value) > 50:
            raise ValueError("sub_goals must not exceed 50 items")

        normalized: List[Union[str, DecomposedTask]] = []

        for entry in value:
            if isinstance(entry, str):
                if stripped := entry.strip():
                    normalized.append(stripped)

                continue

            if isinstance(entry, dict):
                normalized.append(DecomposedTask.model_validate(entry))
                continue

            if isinstance(entry, DecomposedTask):
                normalized.append(entry)
                continue

            raise ValueError(
                f"sub_goals entries must be string or task object, got {type(entry).__name__}"
            )

        if not normalized:
            raise ValueError("sub_goals must contain at least one non-empty entry")

        return normalized
