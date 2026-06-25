from __future__ import annotations

import unittest
from unittest.mock import patch

from fathom.settings.env import FathomSettings


class FathomSettingsCapacityTest(unittest.TestCase):
    """
    Covers environment-backed elevated-capacity settings.
    """

    def test_capacity_fields_load_from_fathom_aliases(self) -> None:
        """
        FATHOM priority environment variables must populate flat capacity fields.
        """

        with patch.dict(
            "os.environ",
            {
                "FATHOM_LLM_PRIORITY_MODE": "adaptive",
                "FATHOM_LLM_PRIORITY_WINDOW": "4",
                "FATHOM_LLM_PRIORITY_FAILURE_THRESHOLD": "2",
                "FATHOM_LLM_PRIORITY_SLOW_THRESHOLD": "3",
                "FATHOM_LLM_PRIORITY_LATENCY_THRESHOLD": "6.5",
                "FATHOM_LLM_PRIORITY_RECOVERY_SUCCESSES": "2",
            },
            clear=True,
        ):
            settings = FathomSettings(_env_file=None)

        self.assertEqual(settings.capacity_mode, "adaptive")
        self.assertEqual(settings.capacity_window, 4)
        self.assertEqual(settings.capacity_failures, 2)
        self.assertEqual(settings.capacity_slows, 3)
        self.assertEqual(settings.capacity_latency, 6.5)
        self.assertEqual(settings.capacity_recovery, 2)

    def test_capacity_fields_load_from_drizz_aliases(self) -> None:
        """
        DRIZZ-prefixed priority variables must work for embedded Fathom runtimes.
        """

        with patch.dict(
            "os.environ",
            {
                "DRIZZ_FATHOM_LLM_PRIORITY_ENABLED": "false",
                "DRIZZ_FATHOM_LLM_PRIORITY_MODE": "adaptive",
            },
            clear=True,
        ):
            settings = FathomSettings(_env_file=None)

        self.assertFalse(settings.capacity_enabled)
        self.assertEqual(settings.capacity_mode, "adaptive")


if __name__ == "__main__":
    unittest.main()
