"""
Validation schemas for screen item pre-validation.

Handles validation requirements extraction from intents, validation
results tracking, and context binding to screen states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field


class ValidationRequirement(BaseModel):
    """
    A single validation requirement to check on screen.

    Attributes:
        item_name: The element or text to validate (e.g., "email field", "submit button")
        description: Optional longer description for context
        match_type: How to match the item - exact match or fuzzy/partial match
        severity: critical=must be present, advisory=log if missing but continue anyway
        optional: If True, missing item doesn't fail validation (just logged)
    """

    item_name: str = Field(..., description="Name/label of the item to validate")
    description: Optional[str] = Field(None, description="Extended description for context")
    match_type: str = Field(
        default="fuzzy",
        description="Matching strategy: 'exact' or 'fuzzy'",
    )
    severity: str = Field(
        default="critical",
        description="Validation severity: 'critical' or 'advisory'",
    )
    optional: bool = Field(
        default=False,
        description="If True, missing item doesn't fail validation",
    )


@dataclass(frozen=True)
class ValidationItemResult:
    """
    Result of validating a single required item.

    Attributes:
        item_name: The item that was validated
        found: Whether the item was found on screen
        confidence: 0.0-1.0 confidence score (higher = more sure)
        details: Additional context (e.g., location, text content matched)
    """

    item_name: str
    found: bool
    confidence: float
    details: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    """
    Complete validation result for a set of requirements against a screen.

    Attributes:
        passed: True if all critical items are present
        items: Per-item validation results
        missing_items: List of critical items that were not found
        found_items: List of items that were successfully detected
        confidence_score: Average confidence across all items (0.0-1.0)
        timestamp: When validation occurred (ISO format)
        notes: Human-readable summary for logging/audit
    """

    passed: bool
    items: List[ValidationItemResult]
    missing_items: List[str]
    found_items: List[str]
    confidence_score: float
    timestamp: str
    notes: str


class ValidationContext:
    """
    Binds validation requirements to current screen capture.
    Used internally by ValidationService to track state across iterations.

    Attributes:
        requirements: List of items to validate
        last_result: Most recent validation result
        retry_count: Number of validation attempts so far
        max_retries: Stop validating after this many attempts
    """

    def __init__(
        self,
        requirements: List[ValidationRequirement],
        max_retries: int = 5,
    ) -> None:
        self.requirements = requirements
        self.last_result: Optional[ValidationResult] = None
        self.retry_count = 0
        self.max_retries = max_retries

    def increment_retry(self) -> None:
        """Track another validation attempt."""
        self.retry_count += 1

    def should_continue_validating(self) -> bool:
        """Check if we should attempt validation again."""
        return self.retry_count < self.max_retries

    def is_satisfied(self) -> bool:
        """Check if current validation result indicates success."""
        return self.last_result is not None and self.last_result.passed
