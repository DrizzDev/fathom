from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.llm import StructuredOutput
from fathom.schemas.localization import VisionLocalizationPayload


class StructuredOutputTest(unittest.TestCase):
    """
    Pins the vendor-neutral structured-output specification consumed by every LLM adapter.
    """

    def test_payload_holds_the_response_model_class(self) -> None:
        """
        The payload field carries the production Pydantic class adapters bind at the SDK boundary.
        """

        specification = StructuredOutput(payload=VisionLocalizationPayload)

        self.assertIs(specification.payload, VisionLocalizationPayload)

    def test_specification_is_immutable(self) -> None:
        """
        Frozen-model guarantee prevents callers from mutating the contract mid-flight.
        """

        specification = StructuredOutput(payload=VisionLocalizationPayload)

        with self.assertRaises(ValidationError):
            specification.payload = VisionLocalizationPayload  # type: ignore[misc]

    def test_payload_must_be_a_base_model_subclass(self) -> None:
        """
        Non-Pydantic types are rejected so adapters can rely on model_json_schema().
        """

        with self.assertRaises(ValidationError):
            StructuredOutput(payload=int)  # type: ignore[arg-type]
