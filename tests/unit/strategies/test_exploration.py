from __future__ import annotations

import unittest
from typing import Optional
from unittest.mock import Mock, patch

from fathom.constants.exploration import DEFAULT_EXPLORATION_INTENT, DISABLED_LOOP_THRESHOLD
from fathom.strategies.exploration import ExplorationStrategy


class TestExplorationStrategy(unittest.TestCase):
    """The exploration strategy configures the agent for deliberate, focused revisits."""

    @staticmethod
    def __build(*, intent: Optional[str] = None) -> None:
        """
        Construct an ExplorationStrategy with mocked ports and an optional focus.
        """

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
            intent=intent,
        )

    @patch("fathom.strategies.exploration.PhaseAnnouncer")
    @patch("fathom.strategies.exploration.GraphContext")
    @patch("fathom.strategies.exploration.AgentState")
    def test_disables_loop_detection(
        self, agent_state_cls: Mock, graph_context_cls: Mock, phase_cls: Mock
    ) -> None:
        """Without a focus, the agent uses the default intent with loop detection disabled."""

        self.__build()

        context = graph_context_cls.return_value
        agent_state_cls.assert_called_once_with(
            intent=DEFAULT_EXPLORATION_INTENT,
            max_steps=7,
            capabilities=context.capabilities,
            loop_threshold=DISABLED_LOOP_THRESHOLD,
        )
        # The loop-disabled agent is installed on the graph context.
        context.set_agent_state.assert_called_once_with(agent_state_cls.return_value)

    @patch("fathom.strategies.exploration.PhaseAnnouncer")
    @patch("fathom.strategies.exploration.GraphContext")
    @patch("fathom.strategies.exploration.AgentState")
    def test_passes_focus_intent_to_context_and_agent_state(
        self, agent_state_cls: Mock, graph_context_cls: Mock, phase_cls: Mock
    ) -> None:
        """A provided focus becomes the intent on both the graph context and the agent."""

        self.__build(intent="Focus on the checkout flow")

        self.assertEqual(graph_context_cls.call_args.kwargs["intent"], "Focus on the checkout flow")
        agent_state_cls.assert_called_once_with(
            intent="Focus on the checkout flow",
            max_steps=7,
            capabilities=graph_context_cls.return_value.capabilities,
            loop_threshold=DISABLED_LOOP_THRESHOLD,
        )


if __name__ == "__main__":
    unittest.main()
