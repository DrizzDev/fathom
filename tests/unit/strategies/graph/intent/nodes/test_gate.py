from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.schemas.actions import Action
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.steps import Step
from fathom.strategies.graph.intent.nodes.gate import ActionGate


class ActionGateTest(unittest.IsolatedAsyncioTestCase):
    """
    Verifies the localization handoff preserves capture context.
    """

    async def test_localize_passes_capture_to_target_localizer(self) -> None:
        """
        Stage-2 localization needs the full capture for ensemble providers.
        """

        capture = ScreenCapture(
            width=100,
            height=200,
            activity="app",
            image=b"image",
            timestamp=1,
        )
        observation = ScreenObservation(
            activity="app",
            elements=(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            hashes=ScreenHashBundle(
                visual_hash="v",
                xml_hash="x",
                interaction_hash="i",
            ),
        )
        step = Step(
            step_number=0,
            screen_hash="v",
            action=Action(
                action_type=ActionType.TAP,
                target="Continue",
                confidence=0.9,
                rationale="Tap Continue",
            ),
        )
        expected = LocalizationResult(
            status=LocalizationStatus.UNRESOLVED,
            confidence=0.0,
            reason="miss",
        )
        context = MagicMock()
        context.target_localizer.localize = AsyncMock(return_value=expected)

        result = await ActionGate(context=context).localize(
            step=step,
            capture=capture,
            observation=observation,
        )

        self.assertIs(result, expected)
        context.target_localizer.localize.assert_awaited_once()
        _, kwargs = context.target_localizer.localize.await_args
        self.assertIs(kwargs["capture"], capture)
