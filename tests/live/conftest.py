from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pytest

_LIVE_TESTS_ENV_FLAG: str = "FATHOM_RUN_LIVE_TESTS"
_LIVE_TESTS_DIRECTORY: Path = Path(__file__).resolve().parent


class LiveTestGuard:
    """
    Collection-time guard that keeps live tests out of the default test run.
    """

    @staticmethod
    def is_enabled() -> bool:
        """
        Return True only when the explicit opt-in env flag is set.
        """

        return os.environ.get(_LIVE_TESTS_ENV_FLAG) == "1"

    @staticmethod
    def is_live_item(*, item: pytest.Item) -> bool:
        """
        Return True only for items whose source file lives under tests/live/.

        Conftest hooks fire at session scope, so the items list is global. Without
        this filter the skip would land on unit tests too, producing a fake-green
        run where every test silently skips.
        """

        try:
            item_path = item.path.resolve()
        except (AttributeError, OSError):
            return False

        try:
            item_path.relative_to(_LIVE_TESTS_DIRECTORY)
        except ValueError:
            return False

        return True


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """
    Skip only items under tests/live/ unless the live-tests env flag is set.
    """

    _ = config
    if LiveTestGuard.is_enabled():
        return

    skip_marker = pytest.mark.skip(
        reason=(
            f"Live tests are opt-in. Set {_LIVE_TESTS_ENV_FLAG}=1 to run them; "
            "they call the real Gemini backend and require credentials."
        ),
    )
    for item in items:
        if LiveTestGuard.is_live_item(item=item):
            item.add_marker(skip_marker)
