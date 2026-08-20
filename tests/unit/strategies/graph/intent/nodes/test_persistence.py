from __future__ import annotations

import unittest

from fathom.strategies.graph.intent.nodes.persistence import GraphStatePersistence


class GraphStatePersistenceShouldSkipLauncherTest(unittest.TestCase):
    """
    Pins :meth:`GraphStatePersistence.should_skip_launcher`.

    The static helper decides whether to persist a step's history entry
    based on the launcher status of its execution and observed packages.
    Steps that both start and end on a launcher app (or end on
    ``unknown``) are skipped — they are not real user-app activity.
    Any launcher → in-app transition or in-app activity must persist
    so the generated script captures the meaningful user journey.
    """

    def test_skips_when_both_endpoints_are_launcher(self) -> None:
        """
        Both endpoints on a launcher package skip persistence.
        """

        self.assertTrue(
            GraphStatePersistence.should_skip_launcher(
                execution_activity="com.android.launcher3/Activity",
                observed_activity="com.android.launcher3/Activity",
            ),
        )

    def test_skips_when_observed_is_unknown(self) -> None:
        """
        Launcher → unknown observed activity is treated as a launcher-only step.
        """

        self.assertTrue(
            GraphStatePersistence.should_skip_launcher(
                execution_activity="com.android.launcher3/Activity",
                observed_activity="unknown/x",
            ),
        )

    def test_persists_when_launcher_transitions_to_app(self) -> None:
        """
        Launcher → in-app activity must persist; user has left the launcher.
        """

        self.assertFalse(
            GraphStatePersistence.should_skip_launcher(
                execution_activity="com.android.launcher3/Activity",
                observed_activity="bundl.delivery.production/Home",
            ),
        )

    def test_persists_when_execution_was_not_on_launcher(self) -> None:
        """
        An execution that did not start on the launcher always persists.
        """

        self.assertFalse(
            GraphStatePersistence.should_skip_launcher(
                execution_activity="bundl.delivery.production/Home",
                observed_activity="bundl.delivery.production/Search",
            ),
        )
