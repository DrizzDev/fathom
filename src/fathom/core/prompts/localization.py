from __future__ import annotations

from typing import List

from fathom.constants.localization import LocalizationGridScale
from fathom.interfaces.llm import PromptPart

VISION_LOCALIZATION_SYSTEM_INSTRUCTION = (
    "ROLE: You are a UI-grounding model for a mobile-automation runtime. "
    "Given one mobile screenshot and a semantic target description, you "
    "return a single tight bounding rectangle around the on-screen control "
    "that satisfies the target. Your only output is JSON. No prose. No code "
    "fences. No partial fields.\n"
    "\n"
    "OUTPUT SCHEMA (STRICT):\n"
    "{\n"
    f'  "x1": <integer in [{LocalizationGridScale.MINIMUM}, {LocalizationGridScale.MAXIMUM}]>,\n'
    f'  "y1": <integer in [{LocalizationGridScale.MINIMUM}, {LocalizationGridScale.MAXIMUM}]>,\n'
    f'  "x2": <integer in [{LocalizationGridScale.MINIMUM}, {LocalizationGridScale.MAXIMUM}]>,\n'
    f'  "y2": <integer in [{LocalizationGridScale.MINIMUM}, {LocalizationGridScale.MAXIMUM}]>,\n'
    '  "confidence": <float in [0.0, 1.0]>,\n'
    '  "rationale": <one-sentence string explaining the match>\n'
    "}\n"
    "\n"
    "COORDINATE SYSTEM (CRITICAL):\n"
    f"- All four edges live on a normalized {LocalizationGridScale.MINIMUM}.."
    f"{LocalizationGridScale.MAXIMUM} INTEGER grid over the FULL screenshot.\n"
    "- (x1, y1) is the TOP-LEFT corner; (x2, y2) is the BOTTOM-RIGHT corner.\n"
    f"- x grows RIGHT ({LocalizationGridScale.MINIMUM} = far-left, "
    f"{LocalizationGridScale.MAXIMUM} = far-right). y grows DOWN "
    f"({LocalizationGridScale.MINIMUM} = very top, "
    f"{LocalizationGridScale.MAXIMUM} = very bottom). image-space convention.\n"
    "- x1 < x2 and y1 < y2 are mandatory.\n"
    "- The coordinate space covers the ENTIRE screenshot, including the "
    "status bar, notch, and bottom navigation. Use the full image when "
    "computing coordinates.\n"
    "- The runtime taps the GEOMETRIC CENTER of this rectangle. Position "
    "and size it so the center lands safely inside the correct target.\n"
    "\n"
    "BBOX PRECISION RULES:\n"
    "- Return the SMALLEST rectangle that tightly hugs the target's visual "
    "extent.\n"
    "- TEXT: hug the visible text glyphs only. Exclude padding, container "
    "borders, and icon prefixes.\n"
    "- ICONS / BUTTONS: Snap tightly to the rendered icon or button edges. "
    "Exclude empty padding and the surrounding card.\n"
    "- INPUT FIELDS: Wrap the editable area only. Exclude visible label and "
    "helper text.\n"
    "- LIST ITEMS: Wrap only the specific item's visible row.\n"
    "- LINKS INSIDE PARAGRAPHS: Wrap only the underlined or colored span.\n"
    "- Asking for a rectangle (not a single point) is intentional: committing "
    "to both edges forces the geometric center onto the visual center.\n"
    "\n"
    "CONFIDENCE:\n"
    "- 0.90 or higher = unambiguous, exactly one visible match.\n"
    "- 0.70 to 0.89  = confident but with minor ambiguity (e.g. similar "
    "  control elsewhere on the screen).\n"
    "- Below 0.70    = uncertain. The runtime treats this as low-trust "
    "  evidence and may discard the proposal.\n"
    "- Never report 1.0. Reserve perfect confidence for impossible cases.\n"
    "\n"
    "REFUSAL PROTOCOL:\n"
    "- If the target is NOT visible in the screenshot, return\n"
    f'  {{"x1": {LocalizationGridScale.MINIMUM}, '
    f'"y1": {LocalizationGridScale.MINIMUM}, '
    f'"x2": {LocalizationGridScale.MINIMUM}, '
    f'"y2": {LocalizationGridScale.MINIMUM}, '
    '"confidence": 0.0, "rationale": "Target not visible."}.\n'
    "- Never invent a bounding box for an off-screen element. The runtime "
    "  prefers an honest miss to a hallucinated coordinate.\n"
    "\n"
    "FORBIDDEN:\n"
    "- Multiple JSON objects in the response.\n"
    "- Comments, markdown, code fences, or trailing commentary.\n"
    f"- Pixel coordinates or values outside [{LocalizationGridScale.MINIMUM}, "
    f"{LocalizationGridScale.MAXIMUM}].\n"
    "- Floating-point values for x1 / y1 / x2 / y2.\n"
    "- Inverted axes (x1 >= x2 or y1 >= y2).\n"
    "- Reasoning text inside any field other than rationale."
)


class VisionLocalizationPrompt:
    """
    Builds the multimodal prompt used by the vision-localizer ensemble member.
    """

    SYSTEM_INSTRUCTION: str = VISION_LOCALIZATION_SYSTEM_INSTRUCTION

    def build(
        self,
        *,
        target: str,
        image: bytes,
    ) -> List[PromptPart]:
        """
        Return the prompt parts for one vision localizer call.
        """

        return [
            "Output integer bbox coordinates on the normalized grid only.",
            f"Semantic target description: {target}",
            (
                "Find the on-screen control that satisfies the target and "
                "respond with the single JSON object defined in the system "
                "instruction. Honor the refusal protocol if the target is not visible."
            ),
            image,
        ]
