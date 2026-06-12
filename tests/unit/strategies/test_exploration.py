from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fathom.constants.exploration import DISABLED_LOOP_THRESHOLD
from fathom.strategies.exploration import ExplorationStrategy


class TestExplorationStrategy(unittest.TestCase):
    """The exploration strategy configures the agent for deliberate revisits."""

    @patch("fathom.strategies.exploration.PhaseAnnouncer")
    @patch("fathom.strategies.exploration.GraphContext")
    @patch("fathom.strategies.exploration.AgentState")
    def test_disables_loop_detection(
        self, agent_state_cls: Mock, graph_context_cls: Mock, phase_cls: Mock
    ) -> None:
        """The agent is configured with loop detection disabled and handed to the context."""

        ExplorationStrategy(
            max_steps=7,
            timeout=0.0,
            workflow_id="wf",
            package_name="com.app",
            seed=None,
            llm=Mock(),
            device=Mock(),
            perception=Mock(),
            memory=Mock(),
            signal=Mock(),
            storage=Mock(),
            telemetry=Mock(),
            path_manager=Mock(),
            configuration=Mock(),
            runtime_configuration=Mock(),
        )

        context = graph_context_cls.return_value
        agent_state_cls.assert_called_once_with(
            intent="Explore application",
            max_steps=7,
            capabilities=context.capabilities,
            loop_threshold=DISABLED_LOOP_THRESHOLD,
        )
        # The loop-disabled agent is installed on the graph context.
        context.set_agent_state.assert_called_once_with(agent_state_cls.return_value)


if __name__ == "__main__":
    unittest.main()
