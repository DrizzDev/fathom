from __future__ import annotations

import io
import unittest
from typing import Tuple
from unittest.mock import AsyncMock

from rich.console import Console

from fathom.adapters.telemetry.console import ConsoleTelemetryAdapter
from fathom.constants.events import FathomEvent


def _build_adapter() -> Tuple[ConsoleTelemetryAdapter, Console]:
    """
    Build a ConsoleTelemetryAdapter whose output we can capture.
    """

    console = Console(file=io.StringIO(), force_terminal=True, width=120)
    adapter = ConsoleTelemetryAdapter(inner=AsyncMock(), console=console)
    return adapter, console


def _captured(console: Console) -> str:
    """
    Return the captured output of the adapter's Console.
    """

    # console.file is the StringIO we passed in.
    return console.file.getvalue()  # type: ignore[union-attr]


class ConsoleTelemetryCueTest(unittest.IsolatedAsyncioTestCase):
    """
    Cover the new cue panel renderers added for demo visibility.
    """

    async def test_intent_classified_simple_renders_green_verdict(self) -> None:
        adapter, console = _build_adapter()
        await adapter.info(
            "Classifier decision",
            type=FathomEvent.INTENT_CLASSIFIED,
            should_decompose=False,
        )
        output = _captured(console)
        self.assertIn("Understanding your request", output)
        self.assertIn("Simple", output)

    async def test_intent_classified_complex_renders_magenta_verdict(self) -> None:
        adapter, console = _build_adapter()
        await adapter.info(
            "Classifier decision",
            type=FathomEvent.INTENT_CLASSIFIED,
            should_decompose=True,
        )
        output = _captured(console)
        self.assertIn("Understanding your request", output)
        self.assertIn("Complex", output)

    async def test_decomposition_complete_renders_numbered_plan(self) -> None:
        adapter, console = _build_adapter()
        await adapter.info(
            "Plan ready",
            type=FathomEvent.DECOMPOSITION_COMPLETE,
            sub_goals=["Open app", "Select tab", "Submit form"],
            decomposed=True,
        )
        output = _captured(console)
        self.assertIn("Plan", output)
        self.assertIn("Open app", output)
        self.assertIn("Select tab", output)
        self.assertIn("Submit form", output)
        # Numbered markers: 1, 2, 3 should all appear.
        for marker in ("1", "2", "3"):
            self.assertIn(marker, output)

    async def test_decomposition_with_empty_list_renders_nothing(self) -> None:
        adapter, console = _build_adapter()
        await adapter.info(
            "Plan ready",
            type=FathomEvent.DECOMPOSITION_COMPLETE,
            sub_goals=[],
        )
        # No plan panel to render for an empty list; output may still
        # contain ambient structlog noise but should NOT mention "Plan".
        output = _captured(console)
        self.assertNotIn("📋 Plan", output)

    async def test_sub_goal_started_panel_has_human_index(self) -> None:
        adapter, console = _build_adapter()
        await adapter.info(
            "Sub-goal started",
            type=FathomEvent.SUB_GOAL_STARTED,
            index=0,  # AgentState is 0-indexed; display should show 1
            total=3,
            description="Open the Strava app",
        )
        output = _captured(console)
        self.assertIn("Starting sub-goal", output)
        self.assertIn("1/3", output)
        self.assertIn("Open the Strava app", output)

    async def test_sub_goal_completed_shows_checkmark(self) -> None:
        adapter, console = _build_adapter()
        await adapter.info(
            "Sub-goal completed",
            type=FathomEvent.SUB_GOAL_COMPLETED,
            index=1,
            total=3,
            description="Tap the Challenges tab",
        )
        output = _captured(console)
        self.assertIn("Completed sub-goal", output)
        self.assertIn("2/3", output)
        self.assertIn("Tap the Challenges tab", output)

    async def test_hitl_requested_renders_loud_yellow_panel(self) -> None:
        adapter, console = _build_adapter()
        await adapter.info(
            "Human input needed",
            type=FathomEvent.HITL_REQUESTED,
            prompt="Should I proceed with checkout?",
        )
        output = _captured(console)
        self.assertIn("Awaiting your input", output)
        self.assertIn("Should I proceed with checkout?", output)

    async def test_inner_adapter_still_receives_forwarded_calls(self) -> None:
        """
        The cue renderers must not cut the inner (structlog) pipeline.
        """

        inner = AsyncMock()
        adapter = ConsoleTelemetryAdapter(inner=inner, console=Console(file=io.StringIO()))
        await adapter.info("x", type=FathomEvent.INTENT_CLASSIFIED, should_decompose=False)
        inner.info.assert_awaited_once_with(
            "x", type=FathomEvent.INTENT_CLASSIFIED, should_decompose=False
        )
