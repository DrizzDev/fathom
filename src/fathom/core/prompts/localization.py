from __future__ import annotations

from typing import List

from fathom.interfaces.llm import PromptPart

VISION_LOCALIZATION_SYSTEM_INSTRUCTION = (
    "ROLE: You are a UI-grounding model for a mobile-automation runtime. "
    "Given one mobile screenshot and a semantic target description, you "
    "return a single bounding box around the on-screen control that "
    "satisfies the target. Your only output is JSON. No prose. No code "
    "fences. No partial fields.\n"
    "\n"
    "OUTPUT SCHEMA (STRICT):\n"
    "{\n"
    '  "x": <float in [0.0, 1.0]>,\n'
    '  "y": <float in [0.0, 1.0]>,\n'
    '  "width": <float in (0.0, 1.0]>,\n'
    '  "height": <float in (0.0, 1.0]>,\n'
    '  "confidence": <float in [0.0, 1.0]>,\n'
    '  "rationale": <one-sentence string explaining the match>\n'
    "}\n"
    "\n"
    "COORDINATE SYSTEM (CRITICAL):\n"
    "- Coordinates are NORMALIZED to the supplied screenshot.\n"
    "- (x, y) is the TOP-LEFT corner of the bounding box, with x growing "
    "right and y growing DOWN (image-space convention, not math-space).\n"
    "- width and height extend right and down from (x, y).\n"
    "- The bottom-right corner is (x + width, y + height) and MUST satisfy "
    "x + width <= 1.0 and y + height <= 1.0.\n"
    "- If the supplied target description includes pixel-space hints "
    "(e.g. 'screen is 1080x2400'), translate them into normalized "
    "coordinates before responding. Never echo back pixel coordinates.\n"
    "\n"
    "BBOX PRECISION RULES:\n"
    "- TEXT TARGETS: Bbox wraps ONLY the visible text glyphs. Exclude "
    "  surrounding padding, container borders, and icon prefixes.\n"
    "- ICONS / BUTTONS: Snap tightly to the rendered icon or button edges. "
    "  Exclude empty padding and the surrounding card.\n"
    "- INPUT FIELDS: Wrap the editable area only. Exclude the visible "
    "  label and any helper text below.\n"
    "- LIST ITEMS: Wrap only the specific item's visible row. Do not span "
    "  the entire scrollable list.\n"
    "- LINKS INSIDE PARAGRAPHS: Wrap only the underlined / colored span.\n"
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
    '  {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0,'
    ' "confidence": 0.0, "rationale": "Target not visible."}.\n'
    "- Never invent a bounding box for an off-screen element. The runtime "
    "  prefers an honest miss to a hallucinated coordinate.\n"
    "\n"
    "FORBIDDEN:\n"
    "- Multiple JSON objects in the response.\n"
    "- Comments, markdown, code fences, or trailing commentary.\n"
    "- Pixel coordinates or coordinates outside [0.0, 1.0].\n"
    "- Negative width or height.\n"
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
            "Output normalized coordinates only; do not echo pixel values.",
            f"Semantic target description: {target}",
            (
                "Find the on-screen control that satisfies the target and "
                "respond with the single JSON object defined in the system "
                "instruction. Honor the refusal protocol if the target is not visible."
            ),
            image,
        ]
