"""
Unit tests for the exploration node factory and graph assembly.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

from fathom.graph.exploration_graph import build_exploration_graph
from fathom.graph.exploration_nodes import ExplorationNodeContext, build_exploration_nodes


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
