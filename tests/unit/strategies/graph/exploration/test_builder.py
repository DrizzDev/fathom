from __future__ import annotations

import unittest
from unittest.mock import Mock

from fathom.constants.graph import NodeName
from fathom.strategies.graph.exploration.builder import ExplorationGraphBuilder


class TestExplorationGraphBuilder(unittest.TestCase):
    """The DFS topology compiles into a valid LangGraph."""

    def test_build_compiles_the_full_phase_machine(self) -> None:
        context = Mock()
        context.configuration.llm.use_cache = True

        compiled = ExplorationGraphBuilder(context=context).build()
        nodes = {str(node) for node in compiled.get_graph().nodes}

        for name in (
            NodeName.GROUND,
            NodeName.BFS_ROUTE,
            NodeName.SCAN,
            NodeName.EXECUTE,
            NodeName.NAVIGATE,
            NodeName.RECORD,
        ):
            self.assertIn(str(name), nodes)


if __name__ == "__main__":
    unittest.main()
