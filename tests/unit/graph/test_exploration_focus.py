"""
Unit tests for focused-exploration intent wiring.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from fathom.graph.exploration_nodes import ExplorationNodeContext
from fathom.prompts.templates import EXPLORATION_FOCUS_DIRECTIVE


class TestFocusedExploration:
    """
    A focus string shapes agent_state.intent; blank or None falls back to the
    default full-breadth intent.
    """

    @staticmethod
    def __context(focus: Optional[str] = None) -> ExplorationNodeContext:
        return ExplorationNodeContext(
            device=MagicMock(),
            capture=MagicMock(),
            vision=MagicMock(),
            knowledge_graph=MagicMock(),
            memory=AsyncMock(),
            focus=focus,
        )

    def test_none_focus_falls_back_to_default_intent(self) -> None:
        ctx = self.__context(focus=None)
        assert "Explore this app" in ctx.agent_state.intent
        assert "Focus on" not in ctx.agent_state.intent

    def test_blank_focus_falls_back_to_default_intent(self) -> None:
        ctx = self.__context(focus="   ")
        assert "Explore this app" in ctx.agent_state.intent

    def test_focus_string_is_embedded_in_intent(self) -> None:
        ctx = self.__context(focus="the checkout flow")
        assert "Focus on exploring the checkout flow" in ctx.agent_state.intent
        assert "skip unrelated sections" in ctx.agent_state.intent

    def test_focus_directive_block_is_present(self) -> None:
        assert "FOCUSED EXPLORATION" in EXPLORATION_FOCUS_DIRECTIVE
        assert "GOAL" in EXPLORATION_FOCUS_DIRECTIVE
