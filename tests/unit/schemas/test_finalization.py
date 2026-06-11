from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.schemas.finalization import HistoryFinalizationBudget


class HistoryFinalizationBudgetScriptBoundsTest(unittest.TestCase):
    """
    The script-finalization timeout must cover real multi-step script
    synthesis on stag without artificially aborting via the historical
    5-second cap. Tests pin the new contract so a future refactor cannot
    silently shrink the budget back below operationally observed minimums.
    """

    def test_default_script_budget_covers_multi_step_synthesis(self) -> None:
        """
        Default of 60 seconds covers the operational p99 plus headroom for
        cancelled-run partial script delivery without holding cancellation indefinitely.
        """

        budget = HistoryFinalizationBudget()

        self.assertEqual(budget.script, 60.0)

    def test_default_flush_budget_remains_short(self) -> None:
        """
        History flush keeps the existing 10-second budget; script generation is the only lifted default.
        """

        budget = HistoryFinalizationBudget()

        self.assertEqual(budget.flush, 10.0)

    def test_upper_bound_accepts_long_synthesis(self) -> None:
        """
        Some runs legitimately need longer than a minute; the upper bound
        must accept values up to 300 seconds so operators can lift the
        ceiling for known-slow workloads without forking the schema.
        """

        budget = HistoryFinalizationBudget(script=300.0)

        self.assertEqual(budget.script, 300.0)

    def test_value_above_upper_bound_rejected(self) -> None:
        """
        Anything above 300 seconds is rejected so misconfigurations
        (e.g. a stray millisecond-to-second conversion) surface at
        load-time instead of hanging a workflow indefinitely.
        """

        with self.assertRaises(ValidationError):
            HistoryFinalizationBudget(script=300.1)


if __name__ == "__main__":
    unittest.main()
