"""Focused-exploration intent wiring.

When ``ExplorationNodeContext`` receives a ``focus`` string, it must shape
``agent_state.intent`` so the cached-prompt ``GOAL`` line tells the VLM to
bias toward the named section. When ``focus`` is ``None`` or blank, the
default full-breadth intent is used.
"""

from unittest.mock import AsyncMock, MagicMock

from fathom.graph.exploration_nodes import ExplorationNodeContext
from fathom.prompts.templates import EXPLORATION_FOCUS_DIRECTIVE


def _make_context(focus=None):
    return ExplorationNodeContext(
        device=MagicMock(),
        capture=MagicMock(),
        vision=MagicMock(),
        knowledge_graph=MagicMock(),
        memory=AsyncMock(),
        focus=focus,
    )


def test_focus_none_falls_back_to_default_intent():
    ctx = _make_context(focus=None)
    assert "Explore this app" in ctx.agent_state.intent
    assert "Focus on" not in ctx.agent_state.intent


def test_blank_focus_falls_back_to_default_intent():
    ctx = _make_context(focus="   ")
    assert "Explore this app" in ctx.agent_state.intent


def test_focus_string_is_embedded_in_agent_intent():
    ctx = _make_context(focus="the checkout flow")
    assert "Focus on exploring the checkout flow" in ctx.agent_state.intent
    assert "skip unrelated sections" in ctx.agent_state.intent


def test_focus_directive_block_exists_and_is_non_empty():
    # The cached system prompt relies on this block being present. If it
    # silently becomes empty the agent loses its bias toward the focus.
    assert "FOCUSED EXPLORATION" in EXPLORATION_FOCUS_DIRECTIVE
    assert "GOAL" in EXPLORATION_FOCUS_DIRECTIVE
