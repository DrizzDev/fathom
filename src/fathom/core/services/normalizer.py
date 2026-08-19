from __future__ import annotations

import re
from typing import Optional


class Normalizer:
    """
    Normalizes freeform text and builds canonical action/validation descriptions with stable grammar.
    """

    __MULTISPACE_RE = re.compile(pattern=r"\s+")
    __GENERIC_TARGET_NAMES = frozenset({"element", "ui element", "none", "label", "unknown"})

    @staticmethod
    def clean(text: Optional[str]) -> str:
        """Normalize whitespace and trim text."""

        if not text:
            return ""

        cleaned = Normalizer.__MULTISPACE_RE.sub(repl=" ", string=str(text)).strip()
        cleaned = re.sub(pattern=r"\s+([,.;:!?])", repl=r"\1", string=cleaned)
        return cleaned

    @staticmethod
    def is_generic_target_name(name: Optional[str]) -> bool:
        """
        Return True when a target name is effectively generic.

        Treats empty/whitespace-only names as generic as well.
        """

        if not name:
            return True

        cleaned = Normalizer.clean(text=name).lower()
        return cleaned in Normalizer.__GENERIC_TARGET_NAMES

    @staticmethod
    def sentence(text: Optional[str]) -> str:
        """Apply lightweight sentence-case normalization."""

        cleaned = Normalizer.clean(text=text)
        if not cleaned:
            return ""

        if cleaned[0].isalpha():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    @staticmethod
    def reasoning(text: Optional[str]) -> str:
        """Normalize freeform reasoning/evidence text without changing semantics."""

        return Normalizer.sentence(text=text)

    @staticmethod
    def action(
        target: str,
        action_type: str,
        text: Optional[str] = None,
        wait_duration: Optional[int | float] = None,
        validation_subject: Optional[str] = None,
    ) -> str:
        """Build canonical action descriptions with stable grammar."""

        kind = Normalizer.clean(text=action_type).lower()
        cleaned_target = Normalizer.clean(text=target)
        if not cleaned_target:
            cleaned_target = "element"

        # For validate actions, prefer the dedicated validation_subject field.
        if kind == "validate" and validation_subject:
            cleaned_subject = Normalizer.clean(text=validation_subject)
            if cleaned_subject:
                cleaned_target = cleaned_subject

        if kind == "tap":
            return f"Tap on {cleaned_target}"

        if kind == "type":
            return f"Type '{Normalizer.clean(text=text)}' into {cleaned_target}"

        if "swipe" in kind:
            direction = kind.split("_")[-1] if "_" in kind else "content"
            return f"Swipe {direction} on {cleaned_target}"

        if kind in ("back", "press_back"):
            return "Press back button"

        if kind in ("home", "press_home"):
            return "Press home button"

        if kind == "wait":
            duration_str = ""
            if wait_duration is not None:
                seconds = (
                    float(wait_duration) / 1000.0 if wait_duration > 100 else float(wait_duration)
                )
                duration_str = f" for {seconds:g} seconds"

            if cleaned_target.lower() == "app to finish loading":
                return f"Wait for the app to finish loading{duration_str}"
            if cleaned_target.lower() == "ad to finish":
                return f"Wait for the ad to finish{duration_str}"
            return f"Wait for {cleaned_target}{duration_str}"

        if kind == "validate":
            return f"Validate {cleaned_target}"

        if kind == "scroll":
            return f"Scroll until you see {cleaned_target}"

        if kind == "long_press":
            return f"Long press on {cleaned_target}"

        if kind == "complete":
            return f"Validate {cleaned_target} (Goal complete)"

        return f"{kind.replace('_', ' ').capitalize()} on {cleaned_target}"

    @staticmethod
    def validation(target: str, *, explicit: bool = False, complete: bool = False) -> str:
        """Build explicit validation descriptions."""

        cleaned_target = Normalizer.clean(text=target) or "Goal State"

        if complete:
            return f"Validate {cleaned_target} (Goal complete)"

        if explicit:
            return f"Validate that {cleaned_target}"

        return f"Validate {cleaned_target}"
