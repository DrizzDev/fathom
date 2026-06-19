from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PerceptionBudget(BaseModel):
    """
    Runtime budget for screen observation enrichment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ocr: int = Field(ge=0, description="Maximum OCR duration in milliseconds.")
    local: int = Field(ge=0, description="Maximum local perception duration in milliseconds.")
    localization: int = Field(ge=0, description="Maximum localization duration in milliseconds.")


class LocalizationBudget(BaseModel):
    """
    Runtime budget for target localization providers.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vision: bool = Field(description="Whether paid vision localization is allowed.")
    attempts: int = Field(ge=0, description="Maximum paid localization calls available.")
    local: int = Field(ge=0, description="Maximum local localization duration in milliseconds.")

    threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum confidence required for local execution.",
    )
