from __future__ import annotations

import unittest

import pytest
from pydantic import ValidationError

from fathom.constants.tools import ToolName
from fathom.schemas.tools import AllowedTools


class AllowedToolsTest(unittest.TestCase):
    """Pins the AllowedTools schema contract."""

    def test_contains_returns_true_for_member(self) -> None:
        """contains() reports membership of declared tool names."""

        tools = AllowedTools(names=frozenset({ToolName.EXECUTE_UI, ToolName.ASK_USER}))

        self.assertTrue(tools.contains(name=ToolName.ASK_USER))
        self.assertTrue(tools.contains(name=ToolName.EXECUTE_UI))

    def test_contains_returns_false_for_non_member(self) -> None:
        """contains() reports absence of undeclared tool names."""

        tools = AllowedTools(names=frozenset({ToolName.EXECUTE_UI}))

        self.assertFalse(tools.contains(name=ToolName.ASK_USER))

    def test_supports_empty_set(self) -> None:
        """An empty allowed set is a valid (degenerate) configuration."""

        tools = AllowedTools(names=frozenset())

        self.assertFalse(tools.contains(name=ToolName.ASK_USER))

    def test_is_immutable(self) -> None:
        """Frozen model prevents reassignment of the names field."""

        tools = AllowedTools(names=frozenset({ToolName.EXECUTE_UI}))

        with pytest.raises(ValidationError):
            tools.names = frozenset()  # type: ignore[misc]
