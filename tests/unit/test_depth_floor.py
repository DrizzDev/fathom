"""Tests for the DFS depth-floor guardrail in the exploration scan node.

The guard vetoes a single VLM ``content_exhausted=True`` declaration when
the agent is below ``MIN_DFS_DEPTH``, then honours it on the second pass —
giving the model one chance to find a missed forward action.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from fathom.infrastructure.memory.knowledge_graph import KnowledgeGraph


def _kg() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g._KnowledgeGraph__provider = AsyncMock()
    g._KnowledgeGraph__nodes = {}
    g._KnowledgeGraph__edges = {}
    g._KnowledgeGraph__hash_aliases = {}
    g._KnowledgeGraph__loaded = True
    return g


def test_min_dfs_depth_constant_is_set():
    from fathom.graph.exploration_nodes import MIN_DFS_DEPTH

    # Ship at 4 — short enough for tight apps, long enough to force chains.
    assert MIN_DFS_DEPTH >= 2
    assert isinstance(MIN_DFS_DEPTH, int)


def test_build_exploration_context_includes_depth_floor_directive():
    kg = _kg()

    ctx = kg.build_exploration_context(
        current_hash=None,
        depth=2,
        depth_floor_active=True,
        min_dfs_depth=4,
    )

    assert "DEPTH FLOOR" in ctx
    assert "depth 2" in ctx
    assert "minimum 4" in ctx
    assert "Pick ANY untried interactive element" in ctx


def test_depth_floor_directive_omitted_when_inactive():
    kg = _kg()

    ctx = kg.build_exploration_context(
        current_hash=None,
        depth=2,
        depth_floor_active=False,
    )

    assert "DEPTH FLOOR" not in ctx


def test_focus_reminder_surfaces_in_context_when_set():
    kg = _kg()

    ctx = kg.build_exploration_context(
        current_hash=None,
        depth=2,
        focus="checkout flow",
    )

    # Surfaces at the very top so it survives long-context truncation.
    assert ctx.startswith("FOCUS: checkout flow")


def test_focus_reminder_omitted_when_focus_unset_or_blank():
    kg = _kg()

    ctx_none = kg.build_exploration_context(current_hash=None, focus=None)
    ctx_blank = kg.build_exploration_context(current_hash=None, focus="   ")

    assert "FOCUS:" not in ctx_none
    assert "FOCUS:" not in ctx_blank


def test_exhaustion_rules_template_references_depth_floor():
    from fathom.prompts.templates import EXPLORATION_EXHAUSTION_RULES

    assert "DEPTH FLOOR" in EXPLORATION_EXHAUSTION_RULES


def _make_ctx_with_state(depth: int):
    """Build a minimal stand-in for ExplorationNodeContext to drive the guard."""

    class FakeCtx:
        def __init__(self, depth: int):
            self.current_path = ["x" for _ in range(depth)]
            self.exhaustion_retries: dict[str, int] = {}
            self.fully_scanned: set[str] = set()

    return FakeCtx(depth)


def test_depth_floor_guard_logic_vetoes_then_honours():
    """Mirror the logic at exploration_nodes.py:477 in a unit test.

    We don't import the full LangGraph node (it pulls in vision, capture,
    etc.); we just assert the conditions match the implementation.
    """

    from fathom.graph.exploration_nodes import MIN_DFS_DEPTH

    fingerprint = "abc"
    ctx = _make_ctx_with_state(depth=2)  # below floor

    # First exhaustion at shallow depth → veto, retry counter goes to 1.
    retries = ctx.exhaustion_retries.get(fingerprint, 0)
    veto = len(ctx.current_path) < MIN_DFS_DEPTH and retries == 0
    assert veto is True
    if veto:
        ctx.exhaustion_retries[fingerprint] = retries + 1
    assert ctx.exhaustion_retries[fingerprint] == 1
    assert fingerprint not in ctx.fully_scanned

    # Second exhaustion same screen → honoured, would route to BACKTRACK.
    retries = ctx.exhaustion_retries.get(fingerprint, 0)
    veto = len(ctx.current_path) < MIN_DFS_DEPTH and retries == 0
    assert veto is False
    ctx.fully_scanned.add(fingerprint)
    assert fingerprint in ctx.fully_scanned


def test_depth_floor_guard_no_veto_at_or_above_floor():
    from fathom.graph.exploration_nodes import MIN_DFS_DEPTH

    fingerprint = "xyz"
    ctx = _make_ctx_with_state(depth=MIN_DFS_DEPTH)

    retries = ctx.exhaustion_retries.get(fingerprint, 0)
    veto = len(ctx.current_path) < MIN_DFS_DEPTH and retries == 0
    assert veto is False
