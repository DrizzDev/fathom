from __future__ import annotations

import json
import unittest
from typing import Any, Dict, Optional

from fathom.adapters.localization.gemini.vision import GeminiVisionLocalizer
from fathom.constants.observation import KeyboardVisibility
from fathom.interfaces.llm import LLMPort
from fathom.schemas.actions import Action, CoordinateSource, CoordinateSystem
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
        structured_output=None,
    ):
        """
        Return the preconfigured :class:`GenerateResult` and increment the
        call counter so callers can verify dispatch happened (or didn't).
        """

        _ = structured_output
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

    async def test_valid_payload_renders_logical_bounds(self) -> None:
        """
        A grid-bbox payload is projected onto the capture's logical dims and stamped LOGICAL with source MODEL.
        """

        llm = _StaticLlm(
            payload={
                "x1": 250,
                "y1": 500,
                "x2": 750,
                "y2": 750,
                "confidence": 0.8,
                "rationale": "tight match",
            },
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
        self.assertIs(proposal.bounds.system, CoordinateSystem.LOGICAL)
        self.assertIs(proposal.bounds.source, CoordinateSource.MODEL)
        self.assertAlmostEqual(proposal.confidence, 0.8)
        self.assertEqual(proposal.source, "gemini.vision")

    async def test_grid_payload_survives_dispatch_translation_on_retina(self) -> None:
        """
        Grid-bbox payload on a 3x retina capture must dispatch at the logical region the localizer named.
        Mislabelling the bounds as DEVICE_PIXEL would divide by the 3x scale and shrink the target ninefold.
        """

        llm = _StaticLlm(
            payload={
                "x1": 344,
                "y1": 465,
                "x2": 656,
                "y2": 518,
                "confidence": 0.9,
                "rationale": "tight overlay button",
            },
        )
        localizer = GeminiVisionLocalizer(llm=llm)

        logical_width = 375
        logical_height = 812
        pixel_width = 1125
        pixel_height = 2436

        proposal = await localizer.locate(
            action=self.__action(target="Alright, got it"),
            capture=self.__capture(width=logical_width, height=logical_height),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertIs(proposal.bounds.system, CoordinateSystem.LOGICAL)

        dispatch_x, dispatch_y, dispatch_width, dispatch_height = (
            proposal.bounds.to_logical_dispatch(
                logical_width=logical_width,
                logical_height=logical_height,
                pixel_width=pixel_width,
                pixel_height=pixel_height,
            )
        )

        self.assertEqual(dispatch_x, 129)
        self.assertEqual(dispatch_y, 377)
        self.assertEqual(dispatch_width, 117)
        self.assertEqual(dispatch_height, 43)

    async def test_all_zero_payload_is_refusal_not_zero_bound_proposal(self) -> None:
        """
        The all-zero coordinate/confidence payload is the refusal protocol
        marker. It must yield ``None`` rather than a degenerate zero-bound
        proposal — the latter would route through the supervisor as a
        valid target and produce no-effect retries.
        """

        llm = _StaticLlm(
            payload={
                "x1": 0,
                "y1": 0,
                "x2": 0,
                "y2": 0,
                "confidence": 0.0,
                "rationale": "Target not visible.",
            },
        )
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
            payload={
                "x1": 100,
                "y1": 100,
                "x2": 600,
                "y2": 600,
                "confidence": 0.9,
                "rationale": "would never be sent",
            },
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
        Payloads whose projected width or height rounds to zero pixels are
        rejected even when the grid coordinates are non-zero. The ensemble
        would otherwise see a zero-area proposal and fail downstream IoU
        clustering with a division-by-zero.
        """

        llm = _StaticLlm(
            payload={
                "x1": 100,
                "y1": 100,
                "x2": 101,
                "y2": 600,
                "confidence": 0.9,
                "rationale": "narrow rectangle for projection rounding",
            },
        )
        localizer = GeminiVisionLocalizer(llm=llm)

        result = await localizer.locate(
            action=self.__action(),
            capture=self.__capture(width=10, height=10),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_confidence_out_of_range_returns_none(self) -> None:
        """
        Provider-reported confidence outside the closed unit interval is
        rejected at the schema boundary. Fail-fast over silent clamp: the
        ensemble falls through to the next member instead of acting on a
        proposal whose self-reported trust signal is unreliable.
        """

        llm = _StaticLlm(
            payload={
                "x1": 100,
                "y1": 100,
                "x2": 600,
                "y2": 600,
                "confidence": 2.5,
                "rationale": "model overshoots the unit interval",
            },
        )
        localizer = GeminiVisionLocalizer(llm=llm)

        proposal = await localizer.locate(
            action=self.__action(),
            capture=self.__capture(),
            budget=self.__budget(),
            observation=self.__observation(),
        )

        self.assertIsNone(proposal)
