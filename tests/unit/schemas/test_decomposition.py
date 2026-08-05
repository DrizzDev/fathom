from __future__ import annotations

import json
import unittest

from google.genai import types as genai_types

from fathom.schemas.decomposition import DecomposedTask, DecompositionSchema
from fathom.schemas.proposal import CaptureProposal, CommandProposal, ObservedProposal
from fathom.schemas.requirement import (
    PressRequirement,
    ScrollRequirement,
    SwipeRequirement,
)


class DecompositionSchemaGeminiCompatibilityTest(unittest.TestCase):
    """
    Pins the decomposition structured-output schema as Gemini-compatible.

    Regression for the on-device failure where a discriminated proposal union emitted JSON-Schema
    ``oneOf``+``discriminator``, which google-genai's ``Schema`` rejects (it accepts ``anyOf`` only).
    Every decomposition ``generate()`` failed schema validation before the model produced anything.
    """

    def test_schema_omits_every_gemini_incompatible_construct(self) -> None:
        """
        The generated JSON schema must avoid all constructs Gemini structured output rejects:
        ``oneOf``/``discriminator`` (discriminated unions), ``exclusiveMinimum``/``exclusiveMaximum``
        (gt/lt bounds), and ``minItems``/``maxItems`` (list-length bounds on union-typed items).
        """

        raw = json.dumps(DecompositionSchema.model_json_schema())
        for forbidden in (
            '"oneOf"',
            '"discriminator"',
            '"exclusiveMinimum"',
            '"exclusiveMaximum"',
            '"minItems"',
            '"maxItems"',
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, raw)
        self.assertIn('"anyOf"', raw)

    def test_gemini_generate_content_config_accepts_the_schema(self) -> None:
        """
        Building the GenerateContentConfig with the schema as response_schema must not raise.
        """

        config = genai_types.GenerateContentConfig(
            response_schema=DecompositionSchema,
            response_mime_type="application/json",
        )
        self.assertIsNotNone(config.response_schema)


class DecomposedTaskProposalParsingTest(unittest.TestCase):
    """
    Pins that each proposal kind parses with the exact canonical enum casing the schema constrains:
    ``kind`` uppercase (SuccessKind), ``operation`` lowercase (ActionType), ``direction`` uppercase.
    """

    def test_observed_proposal_parses(self) -> None:
        """
        An observed proposal validates to ObservedProposal.
        """

        task = DecomposedTask.model_validate(
            {"objective": "Open Settings", "proposal": {"kind": "OBSERVED", "assertion": "open"}}
        )
        self.assertIsInstance(task.proposal, ObservedProposal)

    def test_command_press_proposal_parses_with_lowercase_operation(self) -> None:
        """
        A command proposal with lowercase ``operation`` validates to a PressRequirement.
        """

        task = DecomposedTask.model_validate(
            {
                "objective": "Tap login",
                "proposal": {
                    "kind": "COMMAND",
                    "requirement": {"operation": "tap", "target": "login"},
                    "quote": "Tap login",
                },
            }
        )
        self.assertIsInstance(task.proposal, CommandProposal)
        assert isinstance(task.proposal, CommandProposal)
        self.assertIsInstance(task.proposal.requirement, PressRequirement)

    def test_command_scroll_and_swipe_parse_with_uppercase_direction(self) -> None:
        """
        Scroll/swipe proposals with uppercase ``direction`` validate to the right requirement types.
        """

        scroll = DecomposedTask.model_validate(
            {
                "objective": "Scroll",
                "proposal": {
                    "kind": "COMMAND",
                    "requirement": {"operation": "scroll", "direction": "DOWN"},
                    "quote": "Scroll down",
                },
            }
        )
        swipe = DecomposedTask.model_validate(
            {
                "objective": "Swipe",
                "proposal": {
                    "kind": "COMMAND",
                    "requirement": {"operation": "swipe", "direction": "UP"},
                    "quote": "Swipe up",
                },
            }
        )
        assert isinstance(scroll.proposal, CommandProposal)
        assert isinstance(swipe.proposal, CommandProposal)
        self.assertIsInstance(scroll.proposal.requirement, ScrollRequirement)
        self.assertIsInstance(swipe.proposal.requirement, SwipeRequirement)

    def test_capture_proposal_parses(self) -> None:
        """
        A capture proposal validates to CaptureProposal.
        """

        task = DecomposedTask.model_validate(
            {
                "objective": "Store price",
                "proposal": {
                    "kind": "CAPTURE",
                    "subject": "price",
                    "name": "item_price",
                    "provenance": "USER",
                },
            }
        )
        self.assertIsInstance(task.proposal, CaptureProposal)


if __name__ == "__main__":
    unittest.main()
