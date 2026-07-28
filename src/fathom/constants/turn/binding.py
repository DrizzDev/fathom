from __future__ import annotations

from enum import Enum, IntEnum, StrEnum


class BindingState(StrEnum):
    """
    How firmly a spatial action's target is grounded to an interactive element.
    """

    BOUND = "BOUND"
    MISSING = "MISSING"
    INFERRED = "INFERRED"
    CONTESTED = "CONTESTED"


class BindingOrigin(StrEnum):
    """
    Perception channel that produced the grounded geometry.
    """

    OCR = "OCR"
    VISION = "VISION"
    HYBRID = "HYBRID"
    HIERARCHY = "HIERARCHY"
    ACCESSIBILITY = "ACCESSIBILITY"


class BindingThreshold(float, Enum):
    """
    Confidence bound below which a perceptual grounding degrades to INFERRED.
    """

    CONFIDENCE_FLOOR = 0.5


class BindingLimit(IntEnum):
    """
    Bounds on binding evidence payloads.
    """

    EVIDENCE_PREVIEW = 5
