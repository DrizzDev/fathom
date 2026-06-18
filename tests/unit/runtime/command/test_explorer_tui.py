from __future__ import annotations

import unittest

from fathom.constants.exploration import BFSPhase
from fathom.runtime.command.explorer_tui import STATUS_RUNNING, STATUS_STARTING, ExplorerApp
from fathom.schemas.exploration import ExplorationProgress, TokenUsage


async def _noop_workflow() -> bool:
    return True


class TestExplorerApp(unittest.TestCase):
    """The explorer app applies progress snapshots and tolerates pre-mount activity writes."""

    @staticmethod
    def __app() -> ExplorerApp:
        return ExplorerApp(package="in.swiggy.android", max_steps=25, workflow=_noop_workflow)

    def test_update_progress_populates_state(self) -> None:
        app = self.__app()

        app.update_progress(
            ExplorationProgress(
                step=4,
                max_steps=25,
                phase=BFSPhase.BACKTRACK,
                unique_screens=7,
                coverage=30.0,
                tokens=TokenUsage(prompt=1500, completion=300, cached=200),
                status="exploring",
            )
        )

        snapshot = app._state_snapshot()
        self.assertEqual(snapshot["step"], 4)
        self.assertEqual(snapshot["phase"], "backtrack")
        self.assertEqual(snapshot["unique_screens"], 7)
        self.assertEqual(snapshot["coverage"], 30.0)
        self.assertEqual(snapshot["tokens"], {"prompt": 1500, "completion": 300, "cached": 200})
        self.assertEqual(snapshot["status"], "exploring")
        self.assertEqual(snapshot["status_icon"], STATUS_RUNNING)

    def test_append_activity_is_safe_before_mount(self) -> None:
        app = self.__app()

        # No body widget exists yet; the call must be a no-op, not a crash.
        app.append_activity("navigating to frontier")

        self.assertEqual(app._state_snapshot()["status_icon"], STATUS_STARTING)


if __name__ == "__main__":
    unittest.main()
