from __future__ import annotations

from typing import Final, FrozenSet

MODEL_BOUNDS_MINIMUM_IOU: Final[float] = 0.20
MAX_ACTION_BOUND: Final[int] = 5000
CALL_TO_ACTION_MINIMUM_AREA: Final[int] = 8_000
OVERLAY_MINIMUM_COVERAGE_RATIO: Final[float] = 0.45

VISUAL_CONTROL_MINIMUM_WIDTH: Final[int] = 120
VISUAL_CONTROL_MINIMUM_HEIGHT: Final[int] = 56

VISUAL_CONTROL_MINIMUM_FILL_RATIO: Final[float] = 0.35
VISUAL_CONTROL_MAXIMUM_WIDTH_RATIO: Final[float] = 0.85
VISUAL_CONTROL_MAXIMUM_HEIGHT_RATIO: Final[float] = 0.18


VISUAL_CONTROL_CONFIDENCE: Final[float] = 0.75
VISUAL_CONTROL_MINIMUM_VALUE: Final[int] = 120
VISUAL_CONTROL_MINIMUM_AREA: Final[int] = 8_000
VISUAL_CONTROL_MINIMUM_IOU: Final[float] = 0.35
VISUAL_CONTROL_MINIMUM_SATURATION: Final[int] = 80

OCR_MAXIMUM_TOKEN_LENGTH: Final[int] = 64
OCR_CONFIDENCE_HIGH_FLOOR: Final[float] = 0.85
OCR_CONFIDENCE_MEDIUM_FLOOR: Final[float] = 0.6
OCR_TRIGGER_MIN_MANIFEST_SIZE: Final[int] = 5
OCR_TRIGGER_MIN_TEXT_BEARING_ELEMENTS: Final[int] = 3
OCR_TRIGGER_MANIFEST_TEXT_COVERAGE: Final[float] = 0.3


ICON_MATCH_MINIMUM_SCORE: Final[float] = 0.75
ICON_NON_MAX_SUPPRESSION_IOU: Final[float] = 0.3

ENSEMBLE_IOU_AGREEMENT_FLOOR: Final[float] = 0.5
ENSEMBLE_MIN_AGREEING_MEMBERS: Final[int] = 2
ENSEMBLE_SINGLE_PROPOSAL_CONFIDENCE_FLOOR: Final[float] = 0.9

PHRASE_DENSITY_FLOOR: Final[float] = 0.30
REGIONAL_EVIDENCE_FLOOR: Final[float] = 0.55
CONTAINMENT_MINIMUM_RATIO: Final[float] = 0.50
PHRASE_MATCH_MINIMUM_RECALL: Final[float] = 0.80

ACTION_REGION_HALF_SIDE: Final[int] = 120
ACTION_REGION_STATIC_HAMMING_FLOOR: Final[int] = 4

LABEL_BBOX_AGREEMENT_FLOOR: Final[float] = 0.05

FUSED_WEIGHT_IOU: Final[float] = 0.1
FUSED_WEIGHT_RECALL: Final[float] = 0.5
FUSED_WEIGHT_DENSITY: Final[float] = 0.2
FUSED_WEIGHT_CONTAINMENT: Final[float] = 0.2
FUSED_WEIGHT_SUM_TOLERANCE: Final[float] = 1e-6


VISION_LOCALIZER_ATTEMPTS: Final[int] = 2
VISION_LOCALIZER_TIMEOUT: Final[int] = 30_000
VISION_LOCALIZER_RETRY_BACKOFF: Final[float] = 1.5

VISION_IMAGE_QUALITY: Final[int] = 80
VISION_IMAGE_MAX_DIMENSION: Final[int] = 1536

PIXEL_OVERLAY_MAX_INTENSITY: Final[int] = 90
PIXEL_OVERLAY_MIN_AREA_RATIO: Final[float] = 0.4
PIXEL_OVERLAY_MAX_VARIANCE: Final[float] = 1200.0

CALL_TO_ACTION_TEXT: Final[FrozenSet[str]] = frozenset(
    {
        "ok",
        "done",
        "skip",
        "allow",
        "got it",
        "continue",
        "show results",
    }
)
BUTTON_CLASS_HINTS: Final[FrozenSet[str]] = frozenset(
    {
        "button",
        "visualcontrol",
    }
)
INPUT_CLASS_HINTS: Final[FrozenSet[str]] = frozenset(
    {
        "edittext",
        "textfield",
        "textinput",
        "searchfield",
    }
)
KEYBOARD_CLASS_HINTS: Final[FrozenSet[str]] = frozenset(
    {
        "keyboard",
    }
)
SCROLL_CLASS_HINTS: Final[FrozenSet[str]] = frozenset(
    {
        "tableview",
        "scrollview",
        "recyclerview",
        "collectionview",
    }
)
