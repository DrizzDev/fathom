from __future__ import annotations

from fathom.adapters.localization.vision import VisionLLMLocalizer


class GeminiVisionLocalizer(VisionLLMLocalizer):
    """
    Backward-compatible import for the former Gemini-named localizer.

    The implementation is now provider-neutral and uses whichever ``LLMPort`` is bound by the
    runtime. Existing callers can keep the old import while new composition uses
    :class:`VisionLLMLocalizer` directly.
    """

    @property
    def name(self) -> str:
        """Preserve the legacy ensemble identifier for compatibility callers."""

        return "gemini.vision"
