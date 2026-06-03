from __future__ import annotations

import unittest

from fathom.constants.localization import LocalizationGridScale
from fathom.core.prompts.localization import VisionLocalizationPrompt


class VisionLocalizationPromptTest(unittest.TestCase):
    """
    Covers the multimodal prompt builder for the vision-localizer ensemble member.
    """

    def test_build_omits_screenshot_dimensions(self) -> None:
        """
        Prompt must not announce pixel dimensions because the capture's pixel grid is irrelevant on the normalized grid.
        """

        parts = VisionLocalizationPrompt().build(target="Continue button", image=b"PNGFAKE")

        text_parts = [part for part in parts if isinstance(part, str)]

        for part in text_parts:
            self.assertNotIn("pixels wide", part)
            self.assertNotIn("pixels tall", part)
            self.assertNotIn("Screenshot dimensions", part)

    def test_build_carries_grid_directive(self) -> None:
        """
        The integer-grid directive remains the load-bearing per-call reminder.
        """

        parts = VisionLocalizationPrompt().build(target="Continue button", image=b"PNGFAKE")

        text_parts = [part for part in parts if isinstance(part, str)]
        self.assertTrue(
            any("Output integer bbox coordinates" in part for part in text_parts),
        )

    def test_build_includes_target_and_image(self) -> None:
        """
        Prompt parts must surface the semantic target string and the raw image bytes.
        """

        image = b"PNGFAKE"
        parts = VisionLocalizationPrompt().build(target="Continue button", image=image)

        text_parts = [part for part in parts if isinstance(part, str)]

        self.assertIn(image, parts)
        self.assertTrue(any("Continue button" in part for part in text_parts))

    def test_system_instruction_references_grid_bounds(self) -> None:
        """
        The published system instruction announces the localizer's grid scale verbatim.
        """

        instruction = VisionLocalizationPrompt().SYSTEM_INSTRUCTION

        self.assertIn(str(LocalizationGridScale.MINIMUM), instruction)
        self.assertIn(str(LocalizationGridScale.MAXIMUM), instruction)
