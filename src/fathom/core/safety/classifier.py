from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.safety import UNSAFE_ACTION_KEYWORDS


class IntentSafetyVerdict(BaseModel):
    """
    Outcome of an intent-level safety review performed before workflow start.

    ``safe`` is the executable signal; ``matched_keyword`` is populated only when ``safe`` is
    ``False`` so callers can compose an operator-facing message without re-running the scan.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    safe: bool = Field(description="Whether the intent is cleared for autonomous execution.")
    matched_keyword: Optional[str] = Field(
        default=None,
        description="Token from the unsafe vocabulary that caused the block, when not safe.",
    )


class IntentSafetyClassifier:
    """
    Classifies the user's high-level intent against destructive keywords before the workflow starts.

    Sits at the outer boundary so destructive intents never enter the runtime path; per-step safety
    scanning is not consulted during execution.
    """

    def classify(self, *, intent: str) -> IntentSafetyVerdict:
        """
        Evaluate an intent string and return a typed verdict.

        Matches keywords on regex word boundaries so single-word tokens
        like ``"wipe"`` do not match ``"swipe"`` and multi-word phrases
        like ``"factory reset"`` still match when surrounded by other
        text. The first matched keyword is surfaced for traceability.
        """

        haystack = intent.lower()
        for keyword in UNSAFE_ACTION_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", haystack):
                return IntentSafetyVerdict(safe=False, matched_keyword=keyword)
        return IntentSafetyVerdict(safe=True)
