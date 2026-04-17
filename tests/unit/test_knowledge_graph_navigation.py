"""
Unit tests for KnowledgeGraph query/navigation features.

Tests shortest path finding, cycle detection, reachability analysis, and
related graph traversal capabilities.
"""

from unittest.mock import AsyncMock

import pytest

from fathom.constants import ActionType
from fathom.infrastructure.memory.knowledge_graph import GraphEdge, GraphNode, KnowledgeGraph
from fathom.schemas.actions import Action, Bounds


@pytest.fixture
def knowledge_graph():
    """Create a KnowledgeGraph instance with mocked SQLite provider."""
    kg = KnowledgeGraph()
    kg._KnowledgeGraph__provider = AsyncMock()
    # Pre-populate in-memory graph nodes for testing
    kg._KnowledgeGraph__nodes = {}
    kg._KnowledgeGraph__edges = {}
    kg._KnowledgeGraph__hash_aliases = {}
    kg._KnowledgeGraph__loaded = True
    return kg


def add_test_nodes(kg, node_hashes):
    """Helper to add test nodes to the graph."""
    for hash_val in node_hashes:
        kg._KnowledgeGraph__nodes[hash_val] = GraphNode(
            visual_hash=hash_val,
            activity=f"activity.{hash_val[-2:]}",
            description=f"Screen {hash_val}",
            first_seen=0,
            last_seen=0,
            visit_count=1,
        )


def add_test_edge(kg, source, dest, action_type="tap", action_target="button"):
    """Helper to add a directed edge between two nodes."""
    edge = GraphEdge(
        source_hash=source,
        destination_hash=dest,
        action_type=action_type,
        action_target=action_target,
        count=1,
        first_seen=0,
        last_seen=0,
    )
    kg._KnowledgeGraph__edges.setdefault(source, []).append(edge)


class TestShortestPathFinding:
    """Tests for find_path (shortest path using BFS)."""

    def test_find_path_direct_edge(self, knowledge_graph):
        """Test finding a path when direct edge exists."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b"])
        add_test_edge(kg, "hash_a", "hash_b")

        path = kg.find_path("hash_a", "hash_b")
        assert path is not None
        assert len(path) == 2
        assert path[0] == ("hash_a", None)
        assert path[1][0] == "hash_b"
        assert path[1][1] is not None

    def test_find_path_multi_hop(self, knowledge_graph):
        """Test finding shortest path with multiple hops."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d"])
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_c")
        add_test_edge(kg, "hash_c", "hash_d")

        path = kg.find_path("hash_a", "hash_d")
        assert path is not None
        assert len(path) == 4
        assert path[0][0] == "hash_a"
        assert path[-1][0] == "hash_d"

    def test_find_path_shortest_among_multiple(self, knowledge_graph):
        """Test that BFS finds the shortest path among multiple paths."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d", "hash_e"])
        # Short path: a -> b -> e
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_e")
        # Long path: a -> c -> d -> e
        add_test_edge(kg, "hash_a", "hash_c")
        add_test_edge(kg, "hash_c", "hash_d")
        add_test_edge(kg, "hash_d", "hash_e")

        path = kg.find_path("hash_a", "hash_e")
        assert path is not None
        assert len(path) == 3  # Should be short path (a->b->e)

    def test_find_path_same_node(self, knowledge_graph):
        """Test finding path from node to itself."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a"])

        path = kg.find_path("hash_a", "hash_a")
        assert path is not None
        assert len(path) == 1
        assert path[0] == ("hash_a", None)

    def test_find_path_not_found(self, knowledge_graph):
        """Test when no path exists between nodes."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c"])
        # a -> b, but nothing leads to c
        add_test_edge(kg, "hash_a", "hash_b")

        path = kg.find_path("hash_a", "hash_c")
        assert path is None

    def test_find_path_missing_start_node(self, knowledge_graph):
        """Test when start node doesn't exist."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_b"])

        path = kg.find_path("nonexistent", "hash_b")
        assert path is None

    def test_find_path_max_depth_exceeded(self, knowledge_graph):
        """Test that max_depth prevents infinite exploration."""
        kg = knowledge_graph
        # Create a chain longer than max_depth
        node_count = 15
        node_hashes = [f"hash_{i}" for i in range(node_count)]
        add_test_nodes(kg, node_hashes)

        for i in range(node_count - 1):
            add_test_edge(kg, node_hashes[i], node_hashes[i + 1])

        # Small max_depth should prevent reaching the far nodes
        path = kg.find_path("hash_0", "hash_14", max_depth=5)
        assert path is None


class TestAllPathsFinding:
    """Tests for find_all_paths (all paths up to depth limit)."""

    def test_find_all_paths_single_path(self, knowledge_graph):
        """Test finding all paths when only one exists."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c"])
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_c")

        paths = kg.find_all_paths("hash_a", "hash_c")
        assert len(paths) == 1
        assert len(paths[0]) == 3

    def test_find_all_paths_multiple_paths(self, knowledge_graph):
        """Test finding multiple paths between nodes."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d"])
        # Path 1: a -> b -> d
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_d")
        # Path 2: a -> c -> d
        add_test_edge(kg, "hash_a", "hash_c")
        add_test_edge(kg, "hash_c", "hash_d")

        paths = kg.find_all_paths("hash_a", "hash_d")
        assert len(paths) == 2

    def test_find_all_paths_no_path(self, knowledge_graph):
        """Test finding paths when none exist."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c"])
        add_test_edge(kg, "hash_a", "hash_b")

        paths = kg.find_all_paths("hash_a", "hash_c")
        assert len(paths) == 0

    def test_find_all_paths_respects_max_depth(self, knowledge_graph):
        """Test that max_depth limits path exploration."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d", "hash_e"])
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_c")
        add_test_edge(kg, "hash_c", "hash_d")
        add_test_edge(kg, "hash_d", "hash_e")

        # With max_depth=2, shouldn't be able to reach hash_e
        paths = kg.find_all_paths("hash_a", "hash_e", max_depth=2)
        assert len(paths) == 0


class TestReachability:
    """Tests for reachability analysis."""

    def test_is_reachable_direct_edge(self, knowledge_graph):
        """Test reachability with direct edge."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b"])
        add_test_edge(kg, "hash_a", "hash_b")

        assert kg.is_reachable("hash_a", "hash_b") is True
        assert kg.is_reachable("hash_b", "hash_a") is False

    def test_is_reachable_multi_hop(self, knowledge_graph):
        """Test reachability through multiple hops."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c"])
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_c")

        assert kg.is_reachable("hash_a", "hash_c") is True

    def test_is_reachable_same_node(self, knowledge_graph):
        """Test that a node is reachable from itself."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a"])

        assert kg.is_reachable("hash_a", "hash_a") is True

    def test_get_connected_component(self, knowledge_graph):
        """Test getting all reachable nodes from a start node."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d"])
        # a -> b -> c; a -> d; but nothing outside this
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_c")
        add_test_edge(kg, "hash_a", "hash_d")

        reachable = kg.get_connected_component("hash_a")
        assert reachable == {"hash_a", "hash_b", "hash_c", "hash_d"}

    def test_get_connected_component_isolated_node(self, knowledge_graph):
        """Test connected component for isolated node."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b"])
        # No edges at all

        reachable = kg.get_connected_component("hash_a")
        assert reachable == {"hash_a"}

    def test_get_reverse_connected_component(self, knowledge_graph):
        """Test backward reachability (which nodes can reach a target)."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d"])
        # a -> b; c -> b; d -> b
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_c", "hash_b")
        add_test_edge(kg, "hash_d", "hash_b")

        can_reach = kg.get_reverse_connected_component("hash_b")
        assert can_reach == {"hash_a", "hash_b", "hash_c", "hash_d"}


class TestCycleDetection:
    """Tests for cycle detection."""

    def test_detect_cycles_simple_loop(self, knowledge_graph):
        """Test detecting a simple cycle."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c"])
        # a -> b -> c -> a (cycle)
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_c")
        add_test_edge(kg, "hash_c", "hash_a")

        cycles = kg.detect_cycles()
        assert len(cycles) > 0
        # At least one cycle should include all three nodes
        cycle_found = any(len(cycle) == 4 for cycle in cycles)  # 4 because first and last are same
        assert cycle_found

    def test_detect_cycles_self_loop(self, knowledge_graph):
        """Test detecting self-loops."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a"])
        add_test_edge(kg, "hash_a", "hash_a")

        cycles = kg.detect_cycles()
        assert len(cycles) > 0
        # Self-loop should be detected
        self_loop_found = any(cycle[0] == cycle[-1] == "hash_a" for cycle in cycles)
        assert self_loop_found

    def test_detect_cycles_acyclic_graph(self, knowledge_graph):
        """Test that no cycles are found in acyclic graph."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c"])
        # Simple DAG: a -> b -> c
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_c")

        cycles = kg.detect_cycles()
        assert len(cycles) == 0

    def test_detect_cycles_from_start_node(self, knowledge_graph):
        """Test cycle detection from a specific start node."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d"])
        # Cycle: b -> c -> b; a -> b; d is isolated
        add_test_edge(kg, "hash_b", "hash_c")
        add_test_edge(kg, "hash_c", "hash_b")
        add_test_edge(kg, "hash_a", "hash_b")

        cycles = kg.detect_cycles(start_hash="hash_a")
        assert len(cycles) > 0

    def test_detect_cycles_empty_graph(self, knowledge_graph):
        """Test cycle detection in empty graph."""
        kg = knowledge_graph
        cycles = kg.detect_cycles()
        assert len(cycles) == 0


class TestGraphDiameter:
    """Tests for graph diameter calculation."""

    def test_graph_diameter_simple_chain(self, knowledge_graph):
        """Test diameter of a simple chain."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d"])
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_c")
        add_test_edge(kg, "hash_c", "hash_d")

        diameter = kg.get_graph_diameter()
        assert diameter == 3

    def test_graph_diameter_empty_graph(self, knowledge_graph):
        """Test diameter of empty graph."""
        kg = knowledge_graph
        diameter = kg.get_graph_diameter()
        assert diameter is None

    def test_graph_diameter_disconnected_components(self, knowledge_graph):
        """Test diameter with disconnected components."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c", "hash_d"])
        # Component 1: a -> b
        add_test_edge(kg, "hash_a", "hash_b")
        # Component 2: c -> d (isolated from component 1)
        add_test_edge(kg, "hash_c", "hash_d")

        diameter = kg.get_graph_diameter()
        # Diameter should be 1 (max shortest path within components)
        # or 0 if no paths exist between components
        assert diameter is not None


class TestVisualizationContext:
    """Tests for visualization context generation."""

    def test_get_visualization_context_basic(self, knowledge_graph):
        """Test getting visualization context for a node."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b", "hash_c"])
        add_test_edge(kg, "hash_a", "hash_b", "tap", "button1")
        add_test_edge(kg, "hash_c", "hash_a", "tap", "button2")

        context = kg.get_visualization_context("hash_a")
        assert "node" in context
        assert context["node"]["visual_hash"] == "hash_a"
        assert len(context["outgoing_edges"]) == 1
        assert len(context["inbound_edges"]) == 1

    def test_get_visualization_context_nonexistent(self, knowledge_graph):
        """Test visualization context for nonexistent node."""
        kg = knowledge_graph
        context = kg.get_visualization_context("nonexistent")
        assert context == {}

    def test_get_visualization_context_cycle_detection(self, knowledge_graph):
        """Test that visualization context detects cycles."""
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_a", "hash_b"])
        add_test_edge(kg, "hash_a", "hash_b")
        add_test_edge(kg, "hash_b", "hash_a")  # Create cycle

        context = kg.get_visualization_context("hash_a")
        assert context["in_cycle"] is True


class TestFirstSeenLabelCanonicalization:
    """The first label the LLM uses at a given coord_bucket wins for the session."""

    @pytest.mark.asyncio
    async def test_second_transition_at_same_bucket_adopts_first_label(self, knowledge_graph):
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_src", "hash_dst"])

        first = Action(
            action_type=ActionType.TAP,
            rationale="P3 card, not tried",
            natural_language_target="3rd restaurant card",
            bounds=Bounds(x=500, y=600, width=0, height=0),
        )
        await kg.record_transition(
            source_hash="hash_src", action=first, destination_hash="hash_dst"
        )

        drifted = Action(
            action_type=ActionType.TAP,
            rationale="re-evaluating same card",
            natural_language_target="Joe's Pizza card",  # same element, drifted label
            bounds=Bounds(x=500, y=600, width=0, height=0),
        )
        await kg.record_transition(
            source_hash="hash_src", action=drifted, destination_hash="hash_dst"
        )

        tried = kg.get_tried_actions("hash_src")
        assert len(tried) == 1, "drifted label must merge into first-seen edge"
        action_type, target, bucket, _ = tried[0]
        assert action_type == "tap"
        assert target == "3rd restaurant card"
        assert bucket == "10_12"  # 500//50, 600//50

    @pytest.mark.asyncio
    async def test_different_bucket_creates_separate_edge(self, knowledge_graph):
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_src", "hash_dst"])

        top = Action(
            action_type=ActionType.TAP,
            rationale="top-bar icon",
            natural_language_target="Search icon",
            bounds=Bounds(x=100, y=100, width=0, height=0),
        )
        bottom = Action(
            action_type=ActionType.TAP,
            rationale="bottom-nav tab",
            natural_language_target="Home tab",
            bounds=Bounds(x=100, y=950, width=0, height=0),
        )
        await kg.record_transition(source_hash="hash_src", action=top, destination_hash="hash_dst")
        await kg.record_transition(
            source_hash="hash_src", action=bottom, destination_hash="hash_dst"
        )

        tried = kg.get_tried_actions("hash_src")
        assert len(tried) == 2
        labels = {target for _, target, _, _ in tried}
        assert labels == {"Search icon", "Home tab"}


class TestCategorySamplingCount:
    """count_category_taps must count non-repeatable taps of a given
    element_category on a screen, so the sampling guard can cap long-list
    enumeration."""

    @pytest.mark.asyncio
    async def test_counts_only_sampling_actions_of_matching_category(self, knowledge_graph):
        kg = knowledge_graph
        add_test_nodes(kg, ["hash_src", "hash_card_1", "hash_card_2", "hash_other"])

        def _card(x, y, label):
            return Action(
                action_type=ActionType.TAP,
                rationale="r",
                natural_language_target=label,
                bounds=Bounds(x=x, y=y, width=0, height=0),
                element_category="content_item",
            )

        await kg.record_transition(
            source_hash="hash_src", action=_card(200, 300, "Card A"), destination_hash="hash_card_1"
        )
        await kg.record_transition(
            source_hash="hash_src", action=_card(200, 500, "Card B"), destination_hash="hash_card_2"
        )
        # Different category at a different coord — should NOT count.
        await kg.record_transition(
            source_hash="hash_src",
            action=Action(
                action_type=ActionType.TAP,
                rationale="r",
                natural_language_target="Settings icon",
                bounds=Bounds(x=50, y=50, width=0, height=0),
                element_category="secondary_control",
            ),
            destination_hash="hash_other",
        )
        # A scroll (repeatable) on a content_item — should NOT count.
        await kg.record_transition(
            source_hash="hash_src",
            action=Action(
                action_type=ActionType.SCROLL,
                rationale="r",
                natural_language_target="feed",
                bounds=Bounds(x=500, y=500, width=0, height=0),
                element_category="content_item",
            ),
            destination_hash="hash_other",
        )

        assert kg.count_category_taps("hash_src", "content_item") == 2
        assert kg.count_category_taps("hash_src", "secondary_control") == 1
        assert kg.count_category_taps("hash_src", "primary_action") == 0

    @pytest.mark.asyncio
    async def test_counts_are_aggregated_across_activity_not_just_screen(self, knowledge_graph):
        """A feed that spans multiple visual hashes (different scroll positions)
        must still count as ONE list being sampled — otherwise the LLM could
        cycle across hashes and tap N cards each time."""
        kg = knowledge_graph
        # Two screens on the same activity, different visual hashes (as if the
        # user scrolled and triggered a new hash).
        kg._KnowledgeGraph__nodes["hash_feed_top"] = GraphNode(
            visual_hash="hash_feed_top",
            activity="com.app/FeedActivity",
            description="Feed (top)",
            first_seen=1,
            last_seen=1,
            visit_count=1,
        )
        kg._KnowledgeGraph__nodes["hash_feed_mid"] = GraphNode(
            visual_hash="hash_feed_mid",
            activity="com.app/FeedActivity",
            description="Feed (scrolled)",
            first_seen=2,
            last_seen=2,
            visit_count=1,
        )
        add_test_nodes(kg, ["hash_card_a", "hash_card_b", "hash_card_c"])

        def _card(x, y, label):
            return Action(
                action_type=ActionType.TAP,
                rationale="r",
                natural_language_target=label,
                bounds=Bounds(x=x, y=y, width=0, height=0),
                element_category="content_item",
            )

        # Two cards tapped from the top of the feed, one from the scrolled view.
        await kg.record_transition(
            source_hash="hash_feed_top",
            action=_card(200, 300, "A"),
            destination_hash="hash_card_a",
        )
        await kg.record_transition(
            source_hash="hash_feed_top",
            action=_card(200, 500, "B"),
            destination_hash="hash_card_b",
        )
        await kg.record_transition(
            source_hash="hash_feed_mid",
            action=_card(200, 400, "C"),
            destination_hash="hash_card_c",
        )

        # Per-screen would give 2 and 1 respectively. Per-activity must give 3.
        assert kg.count_category_taps("hash_feed_top", "content_item") == 3
        assert kg.count_category_taps("hash_feed_mid", "content_item") == 3


class TestActivityCoverageSignal:
    """build_exploration_context should push the agent toward unseen activities
    when it lands on one it has already mapped."""

    def _seed(self, kg):
        """Seed the graph with three nodes across two activities."""
        # Two screens on Activity A (the "home" activity), one on Activity B.
        kg._KnowledgeGraph__nodes["hash_home_1"] = GraphNode(
            visual_hash="hash_home_1",
            activity="com.app/HomeActivity",
            description="Home feed",
            first_seen=1,
            last_seen=1,
            visit_count=2,
        )
        kg._KnowledgeGraph__nodes["hash_home_2"] = GraphNode(
            visual_hash="hash_home_2",
            activity="com.app/HomeActivity",
            description="Home feed (scrolled)",
            first_seen=2,
            last_seen=2,
            visit_count=1,
        )
        kg._KnowledgeGraph__nodes["hash_orders"] = GraphNode(
            visual_hash="hash_orders",
            activity="com.app/OrdersActivity",
            description="Orders list",
            first_seen=3,
            last_seen=3,
            visit_count=1,
        )

    def test_known_activities_are_listed(self, knowledge_graph):
        self._seed(knowledge_graph)
        context = knowledge_graph.build_exploration_context(current_hash="hash_home_1")
        # Activity prefix is stripped by normalize_activity.
        assert "KNOWN ACTIVITIES (2):" in context
        assert "HomeActivity" in context
        assert "OrdersActivity" in context

    def test_revisit_warning_fires_when_activity_already_mapped(self, knowledge_graph):
        self._seed(knowledge_graph)
        # hash_home_1 shares its activity with hash_home_2 → revisit signal.
        context = knowledge_graph.build_exploration_context(current_hash="hash_home_1")
        assert "REVISIT" in context
        assert "UNSEEN activity" in context

    def test_no_revisit_warning_on_first_visit_to_activity(self, knowledge_graph):
        self._seed(knowledge_graph)
        # hash_orders is the sole screen on OrdersActivity and visit_count=1.
        context = knowledge_graph.build_exploration_context(current_hash="hash_orders")
        assert "REVISIT" not in context

    def test_no_revisit_warning_when_only_one_activity_mapped(self, knowledge_graph):
        kg = knowledge_graph
        kg._KnowledgeGraph__nodes["hash_only"] = GraphNode(
            visual_hash="hash_only",
            activity="com.app/HomeActivity",
            description="Home feed",
            first_seen=1,
            last_seen=1,
            visit_count=3,
        )
        context = kg.build_exploration_context(current_hash="hash_only")
        # Even revisiting, don't push — there's nowhere else to go yet.
        assert "REVISIT" not in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
