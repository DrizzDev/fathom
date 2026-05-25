from __future__ import annotations

import json
import unittest
from typing import Any, Dict, Optional

from fathom.adapters.localization.gemini.vision import GeminiVisionLocalizer
from fathom.constants.observation import KeyboardVisibility
from fathom.interfaces.llm import LLMPort
from fathom.schemas.actions import Action
from fathom.schemas.budgets import LocalizationBudget
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.results import GenerateResult
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle


class _StaticLlm(LLMPort):
    """
    :class:`LLMPort` test double that returns a frozen payload.

    Production paths read ``model_name`` for cache routing and call
    ``generate`` exactly once per localizer turn. The double records the
    call count so the empty-target test can pin that the localizer never
    reaches the LLM. ``cleanup`` is required by the port contract and is a
    no-op here.
    """

    def __init__(
        self,
        *,
        payload: Optional[Dict[str, Any]] = None,
        raw: Optional[str] = None,
    ) -> None:
        """
        Initialise with either a JSON-serialisable ``payload`` (encoded on
        demand) or a verbatim ``raw`` body. ``raw`` wins when both are set
        — used by the malformed-JSON test.
        """

        self.__payload = payload
        self.__raw = raw
        self.calls: int = 0

    @property
    def model_name(self) -> str:
        """
        Stable identifier surfaced in cache keys and structured logs.
        """

        return "static-llm"

    async def generate(  # type: ignore[no-untyped-def]
        self,
        *,
        use_cache,
        prompt,
        tools=None,
        system_instruction=None,
        conversation_history=None,
    ):
        """
        Return the preconfigured :class:`GenerateResult` and increment the
        call counter so callers can verify dispatch happened (or didn't).
        """

        _ = (prompt, use_cache, system_instruction, tools, conversation_history)
        self.calls += 1
        content = self.__raw if self.__raw is not None else json.dumps(self.__payload)
        return GenerateResult(content=content, metrics={})

    async def cleanup(self) -> None:
        """
        Port-required teardown hook. No resources are held by this double.
        """

        return None


class GeminiVisionLocalizerTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the Gemini vision localizer's payload-decoding and refusal handling.

    The localizer issues exactly one LLM call, parses a normalized
    bounding-box payload, and returns a :class:`LocalizationProposal` —
    or ``None`` for refusals, malformed JSON, blank targets, and zero-area
    payloads. Confidence is clamped to the closed unit interval.
    """

    @staticmethod
    def __capture(*, width: int = 200, height: int = 400) -> ScreenCapture:
        """
        :class:`ScreenCapture` fixture with overridable dimensions so the
        zero-width-pixel test can drive a small canvas where rounding pins
        the rejection path.
        """

        return ScreenCapture(
            width=width,
            height=height,
            activity="app",
            image=b"PNGFAKE",
            timestamp=0,
        )

    @staticmethod
    def __action(*, target: str = "Continue button") -> Action:
        """
        :class:`Action` fixture parameterised on the semantic target. The
        empty-target test passes whitespace to exercise the short-circuit.
        """

        return Action(  # type: ignore[arg-type]
            action_type="tap",
            target=target,
            rationale="test",
            confidence=1.0,
        )

    @staticmethod
    def __budget() -> LocalizationBudget:
        """
        :class:`LocalizationBudget` with paid vision enabled and a generous
        local window so the budget never gates the test.
        """

        return LocalizationBudget(vision=True, attempts=1, local=2000, threshold=0.5)

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Minimal :class:`ScreenObservation` fixture. The vision localizer
        ignores the observation entirely; it exists only to satisfy the
        port signature.
        """

        return ScreenObservation(
            activity="app",
            hashes=ScreenHashBundle(
                visual_hash="0" * 16,
                xml_hash="a" * 16,
                interaction_hash="b" * 16,
            ),
            elements=(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
        )

    async def test_valid_payload_renders_pixel_bounds(self) -> None:
        """
        A well-formed normalized payload renders the bounds into pixel
        space using the capture's width and height, and surfaces the
        proposal with the localizer's canonical source name.
        """

        llm = _StaticLlm(
            payload={"x": 0.25, "y": 0.5, "width": 0.5, "height": 0.25, "confidence": 0.8},
        )
        localizer = GeminiVisionLocalizer(llm=llm)

        proposal = await localizer.locate(
            action=self.__action(),
            capture=self.__capture(width=200, height=400),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.bounds.x, 50)
        self.assertEqual(proposal.bounds.y, 200)
        self.assertEqual(proposal.bounds.width, 100)
        self.assertEqual(proposal.bounds.height, 100)
        self.assertAlmostEqual(proposal.confidence, 0.8)
        self.assertEqual(proposal.source, "gemini.vision")

    async def test_all_zero_payload_is_refusal_not_zero_bound_proposal(self) -> None:
        """
        The all-zero coordinate/confidence payload is the refusal protocol
        marker. It must yield ``None`` rather than a degenerate zero-bound
        proposal — the latter would route through the supervisor as a
        valid target and produce no-effect retries.
        """

        llm = _StaticLlm(payload={"x": 0, "y": 0, "width": 0, "height": 0, "confidence": 0})
        localizer = GeminiVisionLocalizer(llm=llm)

        result = await localizer.locate(
            action=self.__action(),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_malformed_json_response_returns_none(self) -> None:
        """
        A non-JSON response body must not raise. The localizer logs a
        structured ``payload.invalid_json`` event and returns ``None`` so
        the ensemble can poll the next member.
        """

        localizer = GeminiVisionLocalizer(llm=_StaticLlm(raw="not json"))

        result = await localizer.locate(
            action=self.__action(),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_blank_target_short_circuits_before_llm_call(self) -> None:
        """
        A whitespace-only target text must short-circuit before the LLM
        call. The call-count assertion pins that no provider quota is
        spent on requests the model cannot resolve.
        """

        llm = _StaticLlm(
            payload={"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5, "confidence": 0.9},
        )
        localizer = GeminiVisionLocalizer(llm=llm)

        result = await localizer.locate(
            action=self.__action(target="   "),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNone(result)
        self.assertEqual(llm.calls, 0)

    async def test_zero_pixel_box_rejected_as_invalid(self) -> None:
        """
        Payloads whose width or height round to zero pixels after scaling
        are invalid even when the coordinate values are non-zero. The
        ensemble would otherwise see a zero-area proposal and fail
        downstream IoU clustering with a division-by-zero.
        """

        llm = _StaticLlm(
            payload={"x": 0.1, "y": 0.1, "width": 0.0001, "height": 0.5, "confidence": 0.9},
        )
        localizer = GeminiVisionLocalizer(llm=llm)

        result = await localizer.locate(
            action=self.__action(),
            capture=self.__capture(width=10, height=10),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_confidence_clamped_to_unit_interval(self) -> None:
        """
        Provider-reported confidence above 1.0 is clamped at the schema
        boundary so :class:`LocalizationProposal` construction never fails
        validation on out-of-range floats.
        """

        llm = _StaticLlm(
            payload={"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5, "confidence": 2.5},
        )
        localizer = GeminiVisionLocalizer(llm=llm)

        proposal = await localizer.locate(
            action=self.__action(),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.confidence, 1.0)
