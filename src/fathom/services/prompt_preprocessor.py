from __future__ import annotations

import re
from typing import Any, Dict


class PromptPreprocessor:
    """Code-based intent analysis to reduce prompt tokens."""

    @staticmethod
    def extract_hints(intent: str, current_activity: str = "") -> Dict[str, Any]:
        """Extract deterministic info from intent string."""
        hints: Dict[str, Any] = {
            "target_screen": None,
            "typing_text": None,
            "requires_repeat_all": False,
            "element_type": None,
            "is_typing_intent": False,
            "needs_navigation": False,
            "interacted_count": 0,
        }

        # Screen name extraction.
        match = re.search(r'on the ["\']?([^"\']+)["\']?\s*(screen|page)', intent, re.I)
        if match:
            hints["target_screen"] = match.group(1).strip()

        # Typing text extraction.
        match = re.search(r'(?:type|enter|input|fill|write)\s+["\']([^"\']+)["\']', intent, re.I)
        if match:
            hints["typing_text"] = match.group(1)

        # Typing intent detection.
        hints["is_typing_intent"] = bool(
            re.search(r"\b(type|enter|input|fill|write)\b", intent, re.I)
        )

        # Repeat-all detection.
        hints["requires_repeat_all"] = bool(
            re.search(r"\b(every|all|each|all of)\b", intent, re.I)
        )

        # Basic element-type detection.
        element_match = re.search(
            r"(promotional\s+cards?|product\s+cards?|banner\s+ads?|featured\s+items?)",
            intent,
            re.I,
        )
        if element_match:
            hints["element_type"] = element_match.group(1).strip()

        # Navigation detection.
        if (
            hints.get("target_screen")
            and current_activity
            and hints["target_screen"].lower() not in current_activity.lower()
        ):
            hints["needs_navigation"] = True

        return hints

    @staticmethod
    def build_context_prefix(hints: Dict[str, Any]) -> str:
        """Build conditional prompt prefix from hints."""
        parts = []
        if hints.get("typing_text"):
            parts.append(f"[HINT] Text to type: \"{hints['typing_text']}\"")
        if hints.get("needs_navigation"):
            parts.append(f"[HINT] May need to navigate to: {hints['target_screen']}")
        if hints.get("requires_repeat_all"):
            interacted_count = int(hints.get("interacted_count", 0))
            parts.append(
                f"[HINT] Repeat-all intent detected. Continue with untried matches. "
                f"Already interacted: {interacted_count}"
            )
        if hints.get("element_type"):
            parts.append(f"[HINT] Prioritize element type: {hints['element_type']}")
        return "\n".join(parts)
