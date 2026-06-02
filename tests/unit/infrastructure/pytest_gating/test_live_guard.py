from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.live.conftest import LiveTestGuard


_REPO_ROOT: Path = Path(__file__).resolve().parents[4]
_LIVE_DIRECTORY: Path = _REPO_ROOT / "tests" / "live"


class LiveTestGuardEnvFlagTest(unittest.TestCase):
    """
    LiveTestGuard.is_enabled returns True iff the explicit env flag is set to "1".
    """

    def test_is_enabled_returns_true_when_flag_is_one(self) -> None:
        """
        Setting the opt-in env flag to "1" must enable live tests.
        """

        with patch.dict(os.environ, {"FATHOM_RUN_LIVE_TESTS": "1"}):
            self.assertTrue(LiveTestGuard.is_enabled())

    def test_is_enabled_returns_false_when_flag_is_zero(self) -> None:
        """
        Any value other than "1" must keep live tests opt-out.
        """

        with patch.dict(os.environ, {"FATHOM_RUN_LIVE_TESTS": "0"}):
            self.assertFalse(LiveTestGuard.is_enabled())

    def test_is_enabled_returns_false_when_flag_is_unset(self) -> None:
        """
        Missing env flag must keep live tests opt-out (default safe).
        """

        env = {k: v for k, v in os.environ.items() if k != "FATHOM_RUN_LIVE_TESTS"}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(LiveTestGuard.is_enabled())


class LiveTestGuardItemScopingTest(unittest.TestCase):
    """
    is_live_item must classify a pytest Item by its source path: only items
    physically under tests/live/ count as live. This is the regression test
    for the fake-green bug — without scoping, the conftest hook skipped all
    items in the session including unit tests.
    """

    @staticmethod
    def __item_at(*, path: Path) -> MagicMock:
        """
        Build a minimal mock pytest Item with a `.path` attribute.
        """

        item = MagicMock()
        item.path = path
        return item

    def test_item_under_tests_live_is_classified_as_live(self) -> None:
        """
        A real live-test path resolves under tests/live/ and must classify as live.
        """

        live_path = _LIVE_DIRECTORY / "core" / "services" / "qualifier" / "test_llm.py"
        item = self.__item_at(path=live_path)
        self.assertTrue(LiveTestGuard.is_live_item(item=item))

    def test_unit_item_is_not_classified_as_live(self) -> None:
        """
        Any unit-test path must NOT classify as live, even when the live conftest
        is loaded. This prevents the fake-green where every test silently skips.
        """

        unit_path = (
            _REPO_ROOT
            / "tests"
            / "unit"
            / "core"
            / "services"
            / "qualifier"
            / "test_gate.py"
        )
        item = self.__item_at(path=unit_path)
        self.assertFalse(LiveTestGuard.is_live_item(item=item))

    def test_arbitrary_outside_path_is_not_classified_as_live(self) -> None:
        """
        Paths outside the repo must not classify as live.
        """

        item = self.__item_at(path=Path("/tmp/elsewhere/test_foo.py"))
        self.assertFalse(LiveTestGuard.is_live_item(item=item))

    def test_item_without_path_attribute_is_not_classified_as_live(self) -> None:
        """
        A degenerate item without a usable .path must return False, not raise.
        Defensive — pytest can theoretically supply non-file collection items.
        """

        broken = MagicMock()
        del broken.path
        self.assertFalse(LiveTestGuard.is_live_item(item=broken))


if __name__ == "__main__":
    unittest.main()
