from __future__ import annotations

import unittest

from fathom.core.prompts.localization import VisionLocalizationPrompt


class VisionLocalizationPromptTest(unittest.TestCase):
    """
    Covers the multimodal prompt builder for the vision-localizer ensemble member.
    """

    def test_build_omits_screenshot_dimensions(self) -> None:
        """
        Prompt must not announce pixel dimensions because capture dims are logical, not pixel, on retina devices.
        """

        parts = VisionLocalizationPrompt().build(target="Continue button", image=b"PNGFAKE")

        text_parts = [part for part in parts if isinstance(part, str)]
        for part in text_parts:
            self.assertNotIn("pixels wide", part)
            self.assertNotIn("pixels tall", part)
            self.assertNotIn("Screenshot dimensions", part)

    def test_build_keeps_normalized_only_directive(self) -> None:
        """
        The normalized-coordinate directive remains the load-bearing per-call reminder.
        """

        parts = VisionLocalizationPrompt().build(target="Continue button", image=b"PNGFAKE")

        text_parts = [part for part in parts if isinstance(part, str)]
        self.assertTrue(
            any("Output normalized coordinates only" in part for part in text_parts),
        )

    def test_build_includes_target_and_image(self) -> None:
        """
        Prompt parts must surface the semantic target string and the raw image bytes.
        """

        image = b"PNGFAKE"
        parts = VisionLocalizationPrompt().build(target="Continue button", image=image)

        text_parts = [part for part in parts if isinstance(part, str)]
        self.assertTrue(any("Continue button" in part for part in text_parts))
        self.assertIn(image, parts)
