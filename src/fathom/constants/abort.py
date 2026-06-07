from __future__ import annotations

from typing import Final

ABORT_WARMUP_PROMPT: Final[str] = "ok"
DEFAULT_ABORT_DETECTOR_MAX_RETRIES: Final[int] = 2
DEFAULT_ABORT_CONFIDENCE_FLOOR: Final[float] = 0.8
DEFAULT_ABORT_DETECTOR_TIMEOUT: Final[int] = 10_000
DEFAULT_ABORT_DETECTOR_TEMPERATURE: Final[float] = 0.0
DEFAULT_ABORT_FALLBACK_SIMILARITY_FLOOR: Final[float] = 0.85
DEFAULT_ABORT_DETECTOR_MODEL: Final[str] = "gemini-2.5-flash-lite"
