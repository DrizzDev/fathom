"""
Unit tests for KnowledgeGraph.build_exploration_context (depth-floor + focus).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph


class TestBuildExplorationContext:
    """
    The VLM context surfaces the depth-floor directive and the focus reminder.
    """

    @staticmethod
    def __kg() -> KnowledgeGraph:
        return KnowledgeGraph(provider=AsyncMock())

    def test_includes_depth_floor_directive_when_active(self) -> None:
        ctx = self.__kg().build_exploration_context(
            current_hash=None, depth=2, depth_floor_active=True, min_dfs_depth=4
        )
        assert "DEPTH FLOOR" in ctx
        assert "depth 2" in ctx
        assert "minimum 4" in ctx
        assert "Pick ANY untried interactive element" in ctx

    def test_omits_depth_floor_directive_when_inactive(self) -> None:
        ctx = self.__kg().build_exploration_context(
            current_hash=None, depth=2, depth_floor_active=False
        )
        assert "DEPTH FLOOR" not in ctx

    def test_focus_reminder_surfaces_at_top_when_set(self) -> None:
        ctx = self.__kg().build_exploration_context(
            current_hash=None, depth=2, focus="checkout flow"
        )
        # Surfaces at the very top so it survives long-context truncation.
        assert ctx.startswith("FOCUS: checkout flow")

    def test_focus_reminder_omitted_when_unset_or_blank(self) -> None:
        kg = self.__kg()
        assert "FOCUS:" not in kg.build_exploration_context(current_hash=None, focus=None)
        assert "FOCUS:" not in kg.build_exploration_context(current_hash=None, focus="   ")


class TestDepthFloorWiring:
    """
    The depth-floor constant and prompt templates stay in sync.
    """

    def test_min_dfs_depth_constant(self) -> None:
        from fathom.graph.exploration_nodes import MIN_DFS_DEPTH

        assert isinstance(MIN_DFS_DEPTH, int)
        assert MIN_DFS_DEPTH >= 2

    def test_exhaustion_rules_template_references_depth_floor(self) -> None:
        from fathom.prompts.templates import EXPLORATION_EXHAUSTION_RULES

        assert "DEPTH FLOOR" in EXPLORATION_EXHAUSTION_RULES
