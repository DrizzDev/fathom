"""
Unit tests for the exploration node factory and graph assembly.
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.graph.exploration_graph import build_exploration_graph
from fathom.graph.exploration_nodes import (
    DfsNavigator,
    ExplorationNodeContext,
    ExplorationRouter,
    build_exploration_nodes,
)
from fathom.schemas.actions import Action


class TestExplorationNodeFactory:
    """
    The factory exposes every graph node as an async callable.
    """

    @staticmethod
    def __context() -> ExplorationNodeContext:
        return ExplorationNodeContext(
            device=MagicMock(),
            capture=MagicMock(),
            vision=MagicMock(),
            knowledge_graph=MagicMock(),
            memory=AsyncMock(),
        )

    def test_factory_exposes_all_named_nodes(self) -> None:
        nodes = build_exploration_nodes(self.__context())
        assert set(nodes) == {"ground", "bfs_route", "scan", "navigate", "execute", "record"}

    def test_every_node_is_an_async_callable(self) -> None:
        # LangGraph dispatches a node asynchronously only when
        # inspect.iscoroutinefunction is True; a bound async method satisfies
        # this where an instance with an async __call__ would not.
        nodes = build_exploration_nodes(self.__context())
        for name, node in nodes.items():
            assert inspect.iscoroutinefunction(node), name


class TestExplorationGraphAssembly:
    """
    The graph compiles with the class-based node callables.
    """

    @staticmethod
    def __build() -> tuple[object, ExplorationNodeContext]:
        return build_exploration_graph(
            device=MagicMock(),
            capture=MagicMock(),
            vision=MagicMock(),
            knowledge_graph=MagicMock(),
            memory=AsyncMock(),
        )

    def test_graph_compiles(self) -> None:
        compiled, context = self.__build()
        assert compiled is not None
        assert isinstance(context, ExplorationNodeContext)


class TestExplorationRouter:
    """
    Conditional-edge routing decisions for each graph phase.
    """

    @staticmethod
    def __router(*, cancelled: bool = False) -> ExplorationRouter:
        event = asyncio.Event()
        if cancelled:
            event.set()
        context = ExplorationNodeContext(
            device=MagicMock(),
            capture=MagicMock(),
            vision=MagicMock(),
            knowledge_graph=MagicMock(),
            memory=AsyncMock(),
            cancel_event=event,
        )
        return ExplorationRouter(context=context)

    def test_after_ground_proceeds_when_captured(self) -> None:
        assert self.__router().after_ground({"capture": object()}) == "bfs_route"

    def test_after_ground_bails_when_cancelled(self) -> None:
        assert self.__router(cancelled=True).after_ground({"capture": object()}) == "done"

    def test_after_ground_bails_without_capture(self) -> None:
        assert self.__router().after_ground({"capture": None}) == "done"

    def test_after_bfs_route_dispatches_by_phase(self) -> None:
        router = self.__router()
        assert router.after_bfs_route({"bfs_phase": "scan"}) == "scan"
        assert router.after_bfs_route({"bfs_phase": "backtrack"}) == "navigate"

    def test_after_scan_executes_with_action_else_loops(self) -> None:
        router = self.__router()
        assert router.after_scan({"action": object()}) == "execute"
        assert router.after_scan({"action": None}) == "bfs_route"
        assert router.after_scan({"content_exhausted": True}) == "bfs_route"

    def test_after_record_continues_while_running(self) -> None:
        assert self.__router().after_record({}) == "ground"


class TestDfsNavigator:
    """
    Pure BACK/forward navigation computation.
    """

    @staticmethod
    def __action(target: str) -> Action:
        return Action(action_type=ActionType.TAP, rationale="r", target=target)

    def test_forward_only_when_target_extends_current(self) -> None:
        a = self.__action("a")
        b = self.__action("b")
        nav = DfsNavigator.compute_navigation(
            current_path=[("s0", a)], target_path=[("s0", a), ("s1", b)]
        )
        assert [x.action_type for x in nav] == [ActionType.TAP]
        assert nav[0].target == "b"

    def test_back_only_when_target_is_ancestor(self) -> None:
        a = self.__action("a")
        b = self.__action("b")
        nav = DfsNavigator.compute_navigation(current_path=[("s0", a), ("s1", b)], target_path=[])
        assert [x.action_type for x in nav] == [ActionType.BACK, ActionType.BACK]

    def test_ascend_then_descend_on_divergence(self) -> None:
        a = self.__action("a")
        b = self.__action("b")
        c = self.__action("c")
        nav = DfsNavigator.compute_navigation(
            current_path=[("s0", a), ("s1", b)],
            target_path=[("s0", a), ("s2", c)],
        )
        assert nav[0].action_type == ActionType.BACK
        assert nav[-1].target == "c"
        assert len(nav) == 2

    def test_identical_paths_need_no_actions(self) -> None:
        a = self.__action("a")
        nav = DfsNavigator.compute_navigation(current_path=[("s0", a)], target_path=[("s0", a)])
        assert nav == []
