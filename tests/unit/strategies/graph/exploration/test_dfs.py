from __future__ import annotations

import unittest
from typing import List, Optional, Tuple
from unittest.mock import Mock

from fathom.constants import ActionType
from fathom.infrastructure.memory.knowledge_graph import GraphEdge
from fathom.schemas.actions import Action
from fathom.schemas.exploration import BFSQueueEntry
from fathom.strategies.graph.exploration.dfs import DfsNavigator, DfsState


def _action(target: str, action_type: ActionType = ActionType.TAP) -> Action:
    return Action(action_type=action_type, target=target, rationale=f"to {target}")


def _path(*targets: str) -> List[Tuple[str, Action]]:
    return [(f"hash_{target}", _action(target)) for target in targets]


class TestComputeNavigation(unittest.TestCase):
    """The LCA-based BACK-then-forward navigation planner."""

    def test_identical_paths_need_no_actions(self) -> None:
        path = _path("a", "b")
        actions = DfsNavigator.compute_navigation(current_path=path, target_path=list(path))
        self.assertEqual(actions, [])

    def test_disjoint_paths_back_out_then_replay(self) -> None:
        actions = DfsNavigator.compute_navigation(
            current_path=_path("a", "b"),
            target_path=_path("x"),
        )
        # No common prefix: BACK twice out of [a, b], then one forward edge to x.
        self.assertEqual([action.action_type for action in actions[:2]], [ActionType.BACK] * 2)
        self.assertEqual(actions[-1].target, "x")
        self.assertEqual(len(actions), 3)

    def test_shared_prefix_only_diverging_tail_is_renavigated(self) -> None:
        current = _path("a", "b", "c")
        target = _path("a", "d")
        actions = DfsNavigator.compute_navigation(current_path=current, target_path=target)
        # Common ancestor after [a]: BACK out of c and b (2), then forward to d (1).
        back_count = sum(1 for action in actions if action.action_type == ActionType.BACK)
        self.assertEqual(back_count, 2)
        self.assertEqual(actions[-1].target, "d")

    def test_forward_only_from_empty_current(self) -> None:
        actions = DfsNavigator.compute_navigation(current_path=[], target_path=_path("a", "b"))
        self.assertEqual([action.target for action in actions], ["a", "b"])
        self.assertNotIn(ActionType.BACK, [action.action_type for action in actions])


class TestFindOrphanedScreens(unittest.TestCase):
    """Recovery discovery of unscanned screens with a replayable path from root."""

    @staticmethod
    def __knowledge_graph(
        *,
        nodes: List[str],
        paths: Optional[dict[str, Optional[List[Tuple[str, Optional[GraphEdge]]]]]] = None,
        inbound: Optional[dict[str, Optional[Tuple[str, GraphEdge]]]] = None,
    ) -> Mock:
        knowledge_graph = Mock()
        knowledge_graph.nodes = {key: Mock() for key in nodes}
        knowledge_graph.find_path = Mock(
            side_effect=lambda *, end_hash, **_: (paths or {}).get(end_hash)
        )
        knowledge_graph.get_inbound_edge = Mock(
            side_effect=lambda *, destination_hash: (inbound or {}).get(destination_hash)
        )
        return knowledge_graph

    @staticmethod
    def __edge(
        *, source: str, destination: str, action_target: str, action_type: str = "tap"
    ) -> GraphEdge:
        return GraphEdge(
            source_hash=source,
            destination_hash=destination,
            action_type=action_type,
            action_target=action_target,
        )

    def test_returns_only_reachable_unscanned_non_root_screens(self) -> None:
        edge = self.__edge(source="root", destination="child", action_target="Open child")
        knowledge_graph = self.__knowledge_graph(
            nodes=["root", "child", "lonely", "done"],
            paths={"child": [("root", None), ("child", edge)]},
        )
        dfs = DfsState(root_hash="root", fully_scanned={"done"})

        orphans = DfsNavigator(dfs=dfs, knowledge_graph=knowledge_graph).find_orphaned_screens()

        # root excluded; done is fully scanned; lonely has no rooted path or inbound edge.
        self.assertEqual([entry.screen_hash for entry in orphans], ["child"])
        self.assertEqual(orphans[0].parent_hash, "root")
        self.assertEqual(orphans[0].depth, 1)
        self.assertEqual(orphans[0].action_from_parent.action_type, ActionType.TAP)

    def test_builds_full_path_for_deep_frontier_screen(self) -> None:
        first = self.__edge(source="root", destination="mid", action_target="Open mid")
        second = self.__edge(source="mid", destination="leaf", action_target="Open leaf")
        knowledge_graph = self.__knowledge_graph(
            nodes=["root", "leaf"],
            paths={"leaf": [("root", None), ("mid", first), ("leaf", second)]},
        )
        dfs = DfsState(root_hash="root")

        orphans = DfsNavigator(dfs=dfs, knowledge_graph=knowledge_graph).find_orphaned_screens()

        entry = next(item for item in orphans if item.screen_hash == "leaf")
        self.assertEqual(entry.depth, 2)
        self.assertEqual([hop[0] for hop in entry.path_from_root], ["root", "mid"])
        self.assertEqual(entry.parent_hash, "mid")
        self.assertEqual(entry.action_from_parent.target, "Open leaf")

    def test_falls_back_to_inbound_edge_when_no_rooted_path(self) -> None:
        edge = self.__edge(source="src", destination="child", action_target="Open child")
        knowledge_graph = self.__knowledge_graph(nodes=["child"], inbound={"child": ("src", edge)})
        dfs = DfsState(root_hash="root")

        orphans = DfsNavigator(dfs=dfs, knowledge_graph=knowledge_graph).find_orphaned_screens()

        self.assertEqual(orphans[0].parent_hash, "src")
        self.assertEqual(orphans[0].depth, 1)

    def test_unknown_edge_action_type_falls_back_to_tap(self) -> None:
        edge = self.__edge(
            source="root", destination="child", action_target="", action_type="not-a-real-action"
        )
        knowledge_graph = self.__knowledge_graph(
            nodes=["root", "child"], paths={"child": [("root", None), ("child", edge)]}
        )
        dfs = DfsState(root_hash="root")

        orphans = DfsNavigator(dfs=dfs, knowledge_graph=knowledge_graph).find_orphaned_screens()

        self.assertEqual(orphans[0].action_from_parent.action_type, ActionType.TAP)


class TestPathToScreen(unittest.TestCase):
    """Best-effort path reconstruction for a visited screen."""

    def test_root_and_empty_resolve_to_empty_path(self) -> None:
        dfs = DfsState(root_hash="root")
        navigator = DfsNavigator(dfs=dfs, knowledge_graph=Mock(nodes={}))
        self.assertEqual(navigator.path_to_screen(screen_hash="root"), [])
        self.assertEqual(navigator.path_to_screen(screen_hash=None), [])

    def test_queue_entry_path_is_preferred(self) -> None:
        target_path = _path("a", "b")
        dfs = DfsState(root_hash="root")
        dfs.bfs_queue.append(
            BFSQueueEntry(
                screen_hash="target",
                parent_hash="hash_a",
                depth=2,
                action_from_parent=_action("b"),
                path_from_root=target_path,
            )
        )
        navigator = DfsNavigator(dfs=dfs, knowledge_graph=Mock(nodes={}))
        self.assertEqual(navigator.path_to_screen(screen_hash="target"), target_path)

    def test_fallback_drops_last_hop_of_current_path(self) -> None:
        dfs = DfsState(root_hash="root", current_path=_path("a", "b", "c"))
        navigator = DfsNavigator(dfs=dfs, knowledge_graph=Mock(nodes={}))
        self.assertEqual(navigator.path_to_screen(screen_hash="unknown"), _path("a", "b"))


if __name__ == "__main__":
    unittest.main()
