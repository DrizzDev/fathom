from __future__ import annotations

import re
from typing import Optional


class Normalizer:
    """
    Service for normalizing and cleaning natural language text.
    """

    __MULTISPACE_RE = re.compile(pattern=r"\s+")

    @staticmethod
    def clean(text: Optional[str]) -> str:
        """
        Normalize whitespace and trim text.
        """

        if not text:
            return ""

        cleaned = Normalizer.__MULTISPACE_RE.sub(repl=" ", string=str(object=text)).strip()
        cleaned = re.sub(pattern=r"\s+([,.;:!?])", repl=r"\1", string=cleaned)

        return cleaned

    @staticmethod
    def sentence(text: Optional[str]) -> str:
        """
        Apply lightweight sentence-case normalization.
        """

        cleaned = Normalizer.clean(text=text)
        if not cleaned:
            return ""

        if cleaned[0].isalpha():
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    @staticmethod
    def reasoning(text: Optional[str]) -> str:
        """
        Normalize freeform reasoning/evidence text without changing semantics.
        """

        return Normalizer.sentence(text=text)

    @staticmethod
    def wait(condition: Optional[str], rationale: Optional[str] = None) -> Optional[str]:
        """
        Normalize wait condition wording into clear, deterministic phrases.
        """

        cleaned = Normalizer.clean(text=condition)
        if not cleaned:
            return condition

        lower = cleaned.lower()
        rationale_lower = Normalizer.clean(text=rationale).lower()

        if "app to finish loading" in lower or "splash" in lower:
            return "the app is still loading"

        if "first search result" in lower or "search result" in lower:
            return "search results are still loading"

        if "loading indicator" in lower:
            if "search" in rationale_lower or "result" in rationale_lower:
                return "search results are still loading"
            return "content is still loading"

        return cleaned

    @staticmethod
    def action(action_type: str, target: str, text: Optional[str] = None) -> str:
        """
        Build canonical action descriptions with stable grammar.
        """

        kind = Normalizer.clean(text=action_type).lower()
        cleaned_target = Normalizer.clean(text=target) or "UI Element"

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

        if kind == "enter":
            return "Press enter"

        if kind == "wait":
            if cleaned_target.lower() == "app to finish loading":
                return "Wait for the app to finish loading"
            return f"Wait for {cleaned_target}"

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
        """
        Build explicit validation descriptions.
        """

        cleaned_target = Normalizer.clean(text=target) or "Goal State"

        if complete:
            return f"Validate {cleaned_target} (Goal complete)"

        if explicit:
            return f"Validate that {cleaned_target}"

        return f"Validate {cleaned_target}"
