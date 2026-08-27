from __future__ import annotations

import json
import unittest

from fathom.adapters.localization.vision import VisionLLMLocalizer
from fathom.constants.localization import LocalizationGridScale
from fathom.schemas.localization import VisionLocalizationPayload


class VisionPayloadGridClampTest(unittest.TestCase):
    """
    A proposal that overshoots one grid edge still localizes the target, so the edge is clamped
    rather than the whole proposal discarded.
    """

    @staticmethod
    def __within_grid(*, raw: str):
        """
        Invoke the adapter's private grid-clamp over a raw model response.
        """

        return VisionLLMLocalizer._VisionLLMLocalizer__within_grid(  # type: ignore[attr-defined]
            document=json.loads(raw)
        )

    def test_overshooting_edge_is_clamped_and_reported(self) -> None:
        """
        Regression: the real gpt-4.1 response overshot y2 by 50 and was discarded entirely.
        """

        document, clamped = self.__within_grid(
            raw='{"x1":825,"y1":933,"x2":950,"y2":1050,"confidence":0.98,"rationale":"cog icon"}'
        )

        self.assertEqual(clamped, ("y2",))
        self.assertEqual(document["y2"], int(LocalizationGridScale.MAXIMUM))
        self.assertEqual(document["x1"], 825)
        self.assertIsInstance(document["y2"], int)

    def test_clamped_document_passes_the_published_schema(self) -> None:
        """
        The clamped document validates, so the localizer returns a proposal instead of abstaining.
        """

        document, _ = self.__within_grid(
            raw='{"x1":825,"y1":933,"x2":950,"y2":1050,"confidence":0.98,"rationale":"cog icon"}'
        )

        payload = VisionLocalizationPayload.model_validate(document)

        self.assertEqual(payload.y2, int(LocalizationGridScale.MAXIMUM))

    def test_negative_edge_is_clamped_to_the_grid_floor(self) -> None:
        """
        Undershoot is clamped symmetrically with overshoot.
        """

        document, clamped = self.__within_grid(
            raw='{"x1":-20,"y1":10,"x2":950,"y2":900,"confidence":0.5,"rationale":"r"}'
        )

        self.assertEqual(clamped, ("x1",))
        self.assertEqual(document["x1"], int(LocalizationGridScale.MINIMUM))

    def test_in_range_payload_is_untouched(self) -> None:
        """
        A well-formed proposal reports no clamping and keeps every value.
        """

        raw = '{"x1":774,"y1":793,"x2":934,"y2":875,"confidence":0.9,"rationale":"r"}'
        document, clamped = self.__within_grid(raw=raw)

        self.assertEqual(clamped, ())
        self.assertEqual(document, json.loads(raw))

    def test_non_object_response_passes_through_unclamped(self) -> None:
        """
        A response that is not a JSON object is left for schema validation to reject.
        """

        document, clamped = self.__within_grid(raw="[1, 2, 3]")

        self.assertEqual(clamped, ())
        self.assertEqual(document, [1, 2, 3])

    def test_boolean_edge_is_not_treated_as_an_integer(self) -> None:
        """
        ``True`` is an ``int`` in Python; it must not be silently clamped into a coordinate.
        """

        document, clamped = self.__within_grid(
            raw='{"x1":true,"y1":10,"x2":950,"y2":900,"confidence":0.5,"rationale":"r"}'
        )

        self.assertEqual(clamped, ())
        self.assertIs(document["x1"], True)
