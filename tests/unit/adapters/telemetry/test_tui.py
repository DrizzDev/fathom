from __future__ import annotations

import unittest
from typing import List, Tuple

from fathom.adapters.telemetry.tui import TuiTelemetryAdapter
from fathom.constants.exploration import EXPLORATION_PROGRESS_EVENT, BFSPhase
from fathom.schemas.exploration import ExplorationProgress


class _RecordingView:
    """ProgressView stand-in that captures progress snapshots and activity lines."""

    def __init__(self) -> None:
        self.progress: List[ExplorationProgress] = []
        self.activity: List[Tuple[str, str]] = []

    def update_progress(self, progress: ExplorationProgress) -> None:
        self.progress.append(progress)

    def append_activity(self, message: str, *, level: str = "info") -> None:
        self.activity.append((message, level))


class TestTuiTelemetryAdapter(unittest.IsolatedAsyncioTestCase):
    """The adapter turns exploration telemetry into progress and activity updates."""

    async def test_progress_event_updates_the_view(self) -> None:
        view = _RecordingView()
        adapter = TuiTelemetryAdapter(view=view)

        await adapter.info(
            EXPLORATION_PROGRESS_EVENT,
            step=3,
            max_steps=25,
            phase="backtrack",
            unique_screens=8,
            coverage=40.0,
            action="tap Cart",
        )

        self.assertEqual(len(view.progress), 1)
        progress = view.progress[0]
        self.assertEqual(progress.step, 3)
        self.assertEqual(progress.phase, BFSPhase.BACKTRACK)
        self.assertEqual(progress.unique_screens, 8)
        self.assertIn(("step 3: tap Cart", "info"), view.activity)

    async def test_accumulated_tokens_ride_on_progress(self) -> None:
        view = _RecordingView()
        adapter = TuiTelemetryAdapter(view=view)
        adapter.add_tokens(prompt=1000, completion=200, cached=50)
        adapter.add_tokens(prompt=500, completion=100, cached=0)

        await adapter.info(EXPLORATION_PROGRESS_EVENT, step=1, phase="scan")

        tokens = view.progress[0].tokens
        self.assertEqual((tokens.prompt, tokens.completion, tokens.cached), (1500, 300, 50))

    async def test_other_info_becomes_activity(self) -> None:
        view = _RecordingView()
        adapter = TuiTelemetryAdapter(view=view)

        await adapter.info("launching package", package="in.swiggy.android")

        self.assertEqual(len(view.progress), 0)
        self.assertEqual(len(view.activity), 1)
        message, level = view.activity[0]
        self.assertEqual(level, "info")
        self.assertIn("launching package", message)

    async def test_error_is_flagged(self) -> None:
        view = _RecordingView()
        adapter = TuiTelemetryAdapter(view=view)

        await adapter.error("device offline")

        self.assertEqual(view.activity[0][1], "error")

    async def test_invalid_phase_falls_back_to_scan(self) -> None:
        view = _RecordingView()
        adapter = TuiTelemetryAdapter(view=view)

        await adapter.info(EXPLORATION_PROGRESS_EVENT, step=1, phase="bogus")

        self.assertEqual(view.progress[0].phase, BFSPhase.SCAN)


if __name__ == "__main__":
    unittest.main()
