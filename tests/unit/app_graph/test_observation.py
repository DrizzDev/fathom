"""Exploration navigation must point to an observed screen."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fathom.constants import ActionType
from fathom.constants.state import CommonStateKey
from fathom.schemas.actions import Action
from fathom.schemas.exploration import ExplorationGraph
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step, StepResult
from fathom.strategies.graph.exploration.nodes import ExplorationNodeProvider
from fathom.strategies.graph.exploration.state import ExplorationGraphState


class TestObservation(unittest.IsolatedAsyncioTestCase):
    async def test_success_targets_the_next_observed_capture(self) -> None:
        await self.__check(success=True, expected=1)

    async def test_failed_action_does_not_create_navigation(self) -> None:
        await self.__check(success=False, expected=0)

    async def test_first_capture_has_no_incoming_edge(self) -> None:
        await self.__check(success=True, expected=0, previous=False)

    async def test_unknown_origin_has_no_navigation(self) -> None:
        await self.__check(success=True, expected=0, origin="0")

    async def __check(
        self, *, success: bool, expected: int, previous: bool = True, origin: str = "before"
    ) -> None:
        capture = ScreenCapture(
            width=100, height=200, activity="Search", image=b"image", timestamp=1
        )
        graph = ExplorationGraph()
        graph.add_screen(
            ScreenState(activity="Home", activity_hash="home", visual_hash="before", timestamp=0)
        )
        context = SimpleNamespace(
            is_cancelled=False,
            workflow_id="run",
            catalog=Mock(),
            applications=None,
            exploration_graph=graph,
            agent_state=SimpleNamespace(step_count=1, update_screen=Mock(return_value=True)),
            perception=SimpleNamespace(
                perceive=AsyncMock(return_value=capture),
                compute_visual_hash=Mock(return_value="after"),
            ),
        )
        result = StepResult(
            step=Step(
                action=Action(action_type=ActionType.BACK, rationale="Return to previous screen"),
                screen_hash="before",
                step_number=1,
            ),
            success=success,
            pre_hash=origin,
            post_hash="0",
            screen_changed=True,
            duration=1,
        )
        state = ExplorationGraphState({CommonStateKey.STEP_RESULT: result} if previous else {})
        provider = ExplorationNodeProvider(context=context)
        await provider.ground(state)
        self.assertIn("after", graph.nodes)
        self.assertEqual(len(graph.edges), expected)
        if expected:
            self.assertEqual(graph.edges[0][0], "before")
            self.assertEqual(graph.edges[0][2], "after")
        self.assertFalse(any(edge[2] == "0" for edge in graph.edges))
