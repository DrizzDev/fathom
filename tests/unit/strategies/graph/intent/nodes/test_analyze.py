from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, Mock

from fathom.constants import ActionType
from fathom.constants.retries import RetryKind
from fathom.constants.state import (
    TERMINAL_COMPLETION_REASONS,
    CommonStateKey,
    CompletionReason,
    IntentStateKey,
    VerifyMode,
)
from fathom.constants.tools import StateNamespace
from fathom.core.agent.state import AgentState
from fathom.core.exceptions import ToolValidationError
from fathom.core.services.timing import RunClock
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.results import (
    AnalysisResult,
    PlanContext,
    PlanResult,
    PlanTurn,
    ToolErrorFeedback,
)
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.tools import StateUpdate
from fathom.strategies.graph.intent.nodes.analyze import AnalyzeNode
from tests.builders.agent import AgentFixtures
from tests.builders.subgoals import SubGoalFixtures


class _Persistence:
    def __init__(self) -> None:
        self.last: Dict[Any, Any] = {}

    def restore(self, *, state: Dict[Any, Any]) -> None:
        _ = state

    def persist(self, *, result: Dict[Any, Any]) -> None:
        self.last = dict(result)


class AnalyzeNodeFailureBoundaryTest(unittest.IsolatedAsyncioTestCase):
    """
    Covers ANALYZE failure boundaries.
    """

    @staticmethod
    def __provider(
        *,
        agent_state: AgentState,
        planner: Optional[Mock] = None,
        cancelled: bool = False,
        max_steps: int = 10,
    ) -> SimpleNamespace:
        """
        Build the AnalyzeNode provider surface used by failure-boundary tests.
        """

        return SimpleNamespace(
            is_cancelled=AsyncMock(return_value=cancelled),
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            context=SimpleNamespace(
                intent="test-intent",
                phase=AsyncMock(),
                workflow_id="run-test",
                max_steps=max_steps,
                agent_state=agent_state,
                context_manager=AgentFixtures.context_manager(),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(100, 200))),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
                planner=planner or Mock(),
                use_xml=True,
                reasoner=Mock(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                clock=RunClock(),
                memory=SimpleNamespace(set=AsyncMock()),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
            ),
        )

    async def test_planner_exception_terminates_instead_of_retrying_forever(self) -> None:
        """
        Deterministic planner failures must fail fast instead of returning
        ``SHOULD_RETRY=True`` and creating a graph loop.
        """

        agent_state = AgentState(
            intent="finish onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        planner = Mock()
        planner.plan_step = AsyncMock(side_effect=ValueError("bad planner state" * 80))
        provider = self.__provider(agent_state=agent_state, planner=planner)
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=100,
                    height=200,
                    activity="app",
                    image=b"png",
                    timestamp=1,
                )
            }
        )

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.FAILED.value)
        self.assertIn("bad planner state", result[CommonStateKey.FAILURE_DIAGNOSTIC])
        self.assertLessEqual(len(result[CommonStateKey.FAILURE_DIAGNOSTIC]), 500)
        self.assertEqual(
            provider.persistence.last[CommonStateKey.COMPLETION_REASON],
            CompletionReason.FAILED.value,
        )

    async def test_cancelled_analysis_clears_retry_and_deferral_state(self) -> None:
        """
        ANALYZE cancellation must not leave retry or complete-deferral state alive.
        """

        agent_state = AgentState(
            intent="finish onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.record_complete_deferral()
        provider = self.__provider(agent_state=agent_state, cancelled=True)
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={})  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.CANCELLED.value)
        self.assertEqual(agent_state.consecutive_complete_deferrals, 0)

    async def test_pre_planning_max_steps_clears_retry_and_deferral_state(self) -> None:
        """
        The pre-planning max-step guard must produce a fully terminal graph patch.
        """

        agent_state = AgentState(
            intent="finish onboarding",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.record_complete_deferral()
        agent_state.record_step(
            result=StepResult(
                step=Step(
                    action=Action(
                        action_type=ActionType.VALIDATE,
                        target="current screen",
                        rationale="validate state",
                    ),
                    step_number=0,
                    screen_hash="screen",
                ),
                success=True,
                duration=1,
                pre_hash="screen",
                post_hash="screen",
                screen_changed=False,
            )
        )
        provider = self.__provider(agent_state=agent_state, max_steps=1)
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(state={})  # type: ignore[arg-type]

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertFalse(result[IntentStateKey.SHOULD_RETRY])
        self.assertEqual(result[CommonStateKey.COMPLETION_REASON], CompletionReason.MAX_STEPS.value)
        self.assertEqual(agent_state.consecutive_complete_deferrals, 0)

    async def test_state_change_only_plan_routes_to_ground_without_step(self) -> None:
        """
        A non-command tool response writes state but never creates a planned step.
        """

        agent_state = AgentState(
            intent="remember selected price",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        planner = Mock()
        planner.plan_step = AsyncMock(
            return_value=PlanTurn(
                plan=PlanResult(
                    step=None,
                    is_complete=False,
                    should_retry=True,
                    reason="Remember selected price",
                    updates=(
                        StateUpdate(
                            namespace=StateNamespace.MEMORY,
                            key="item_price",
                            value="94",
                        ),
                    ),
                )
            )
        )
        provider = self.__provider(agent_state=agent_state, planner=planner)
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=100,
                    height=200,
                    activity="app",
                    image=b"png",
                    timestamp=1,
                )
            }
        )

        provider.context.memory.set.assert_awaited_once_with(key="item_price", value="94")
        self.assertFalse(result[CommonStateKey.IS_COMPLETE])
        self.assertTrue(result[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(result[IntentStateKey.PLANNED_STEP])

    async def test_tool_schema_validation_exhaustion_routes_to_retry_not_failed(
        self,
    ) -> None:
        """
        Exhausted tool-schema retries consume planner budget instead of marking the run failed.
        """

        agent_state = AgentState(
            intent="search product",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        planner = Mock()
        planner.plan_step = AsyncMock(
            side_effect=ToolValidationError(
                ToolErrorFeedback(
                    tool_name="execute_ui",
                    tool_call_id=None,
                    error_kind="validation",
                    message=(
                        "execute_ui arguments validation failed at action: "
                        "validation_subject is required for action_type='validate'."
                    ),
                )
            )
        )
        provider = self.__provider(agent_state=agent_state, planner=planner)
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]
        state: Dict[Any, Any] = {
            CommonStateKey.CAPTURE: ScreenCapture(
                width=100,
                height=200,
                activity="app",
                image=b"png",
                timestamp=1,
            )
        }

        result = await node.run(state=state)

        self.assertFalse(result[CommonStateKey.IS_COMPLETE])
        self.assertTrue(result[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(result[IntentStateKey.PLANNED_STEP])
        self.assertEqual(agent_state.retries.planner.count, 1)
        attempt = agent_state.last_retry_attempt
        assert attempt is not None
        self.assertEqual(attempt.kind, RetryKind.LLM_FEEDBACK)
        diagnostic = result[CommonStateKey.FAILURE_DIAGNOSTIC]
        assert isinstance(diagnostic, str)
        self.assertIn("validation_subject", diagnostic)


class AnalyzeNodeScreenResolutionTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins that the planner receives capture dims, not stale device dims.
    """

    async def test_planner_receives_capture_dims_not_device_dims(self) -> None:
        """
        Landscape capture must produce landscape screen dims in the planner call.
        """

        agent_state = AgentState(
            intent="open the app",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        planner = Mock()
        planner.plan_step = AsyncMock(
            return_value=SimpleNamespace(
                plan=SimpleNamespace(
                    action=None,
                    metrics=None,
                    rationale="",
                    step=None,
                    is_complete=False,
                    reasoning="",
                    ux_label="",
                    use_xml_grounding=True,
                ),
                events=(),
            ),
        )
        provider = SimpleNamespace(
            is_cancelled=AsyncMock(return_value=False),
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            context=SimpleNamespace(
                intent="test-intent",
                phase=AsyncMock(),
                workflow_id="run-test",
                max_steps=10,
                agent_state=agent_state,
                context_manager=AgentFixtures.context_manager(),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(1080, 2340))),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
                planner=planner,
                use_xml=True,
                reasoner=Mock(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                clock=RunClock(),
                memory=SimpleNamespace(set=AsyncMock()),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
            ),
        )
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=2340,
                    height=1080,
                    activity="app",
                    image=b"png",
                    timestamp=1,
                )
            }
        )

        call = planner.plan_step.await_args
        self.assertIsNotNone(call)
        kwargs = call.kwargs
        self.assertEqual(kwargs["screen_width"], 2340)
        self.assertEqual(kwargs["screen_height"], 1080)


class AnalyzeNodeVerifyModeTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins VERIFY_MODE stamping for AnalyzeNode completion routes.
    """

    async def test_final_active_subgoal_completion_routes_pending_final_commit(self) -> None:
        """
        AnalyzeNode must stamp PENDING_FINAL_COMMIT when it routes an active final sub-goal to VERIFY.
        """

        agent_state = AgentState(
            intent="change the address to finance office",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.set_sub_goals(
            [SubGoalFixtures.make(index=0, description="Confirm Finance address")]
        )
        planner = Mock()
        planner.plan_step = AsyncMock(
            return_value=SimpleNamespace(
                plan=SimpleNamespace(
                    metrics=None,
                    step=None,
                    is_complete=True,
                    should_retry=False,
                    reason="Address appears selected",
                    context=PlanContext(),
                ),
                events=(),
            ),
        )
        provider = SimpleNamespace(
            is_cancelled=AsyncMock(return_value=False),
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            context=SimpleNamespace(
                intent="test-intent",
                phase=AsyncMock(),
                workflow_id="run-test",
                max_steps=10,
                agent_state=agent_state,
                context_manager=AgentFixtures.context_manager(),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(100, 200))),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
                planner=planner,
                use_xml=True,
                reasoner=Mock(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                clock=RunClock(),
                memory=SimpleNamespace(set=AsyncMock()),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
            ),
        )
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=100,
                    height=200,
                    activity="app",
                    image=b"png",
                    timestamp=1,
                )
            }
        )

        self.assertTrue(result[CommonStateKey.IS_COMPLETE])
        self.assertEqual(
            result[IntentStateKey.VERIFY_MODE],
            VerifyMode.PENDING_FINAL_COMMIT.value,
        )

    async def test_deferred_completion_clears_planned_step_and_forces_retry(self) -> None:
        """
        A deferred completion claim must replan instead of executing a stale step attached to the complete verdict.
        """

        agent_state = AgentState(
            intent="change the address to finance office",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.set_sub_goals(
            [
                SubGoalFixtures.make(index=0, description="Tap address selector"),
                SubGoalFixtures.make(index=1, description="Confirm Finance address"),
            ]
        )
        planner = Mock()
        planner.plan_step = AsyncMock(
            return_value=SimpleNamespace(
                plan=SimpleNamespace(
                    metrics=None,
                    is_complete=True,
                    should_retry=False,
                    reason=CompletionReason.SUCCESS.value,
                    context=PlanContext(),
                    step=Step(
                        action=Action(
                            action_type=ActionType.TAP,
                            target="stale target",
                            rationale="stale action attached to completion claim",
                        ),
                        step_number=0,
                        screen_hash="screen",
                    ),
                ),
                events=(),
            ),
        )
        provider = SimpleNamespace(
            is_cancelled=AsyncMock(return_value=False),
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            context=SimpleNamespace(
                intent="test-intent",
                phase=AsyncMock(),
                workflow_id="run-test",
                max_steps=10,
                agent_state=agent_state,
                context_manager=AgentFixtures.context_manager(),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(100, 200))),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
                planner=planner,
                use_xml=True,
                reasoner=Mock(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                clock=RunClock(),
                memory=SimpleNamespace(set=AsyncMock()),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
            ),
        )
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=100,
                    height=200,
                    activity="app",
                    image=b"png",
                    timestamp=1,
                )
            }
        )

        self.assertFalse(result[CommonStateKey.IS_COMPLETE])
        self.assertTrue(result[IntentStateKey.SHOULD_RETRY])
        self.assertIsNone(result[IntentStateKey.PLANNED_STEP])
        self.assertIsNone(result[CommonStateKey.COMPLETION_REASON])

    async def test_terminal_completion_does_not_stamp_verify_mode(self) -> None:
        """
        Terminal failure reasons route to END and must clear VERIFY_MODE.
        """

        agent_state = AgentState(
            intent="change the address to finance office",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        for reason in TERMINAL_COMPLETION_REASONS:
            with self.subTest(reason=reason):
                planner = Mock()
                planner.plan_step = AsyncMock(
                    return_value=SimpleNamespace(
                        plan=SimpleNamespace(
                            metrics=None,
                            step=None,
                            is_complete=True,
                            should_retry=False,
                            reason=reason,
                            context=PlanContext(),
                        ),
                        events=(),
                    ),
                )
                provider = SimpleNamespace(
                    is_cancelled=AsyncMock(return_value=False),
                    persistence=_Persistence(),
                    hitl=SimpleNamespace(prompt=AsyncMock()),
                    context=SimpleNamespace(
                        intent="test-intent",
                        phase=AsyncMock(),
                        workflow_id="run-test",
                        max_steps=10,
                        agent_state=agent_state,
                        context_manager=AgentFixtures.context_manager(),
                        device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(100, 200))),
                        signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                        configuration=SimpleNamespace(
                            intent=SimpleNamespace(prompt_user_if_stuck=False)
                        ),
                        planner=planner,
                        use_xml=True,
                        reasoner=Mock(),
                        metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                        clock=RunClock(),
                        memory=SimpleNamespace(set=AsyncMock()),
                        telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
                    ),
                )
                node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

                result = await node.run(
                    state={
                        CommonStateKey.CAPTURE: ScreenCapture(
                            width=100,
                            height=200,
                            activity="app",
                            image=b"png",
                            timestamp=1,
                        )
                    }
                )

                self.assertTrue(result[CommonStateKey.IS_COMPLETE])
                self.assertIsNone(result[IntentStateKey.VERIFY_MODE])


class AnalyzeNodeAnalysisStatePublicationTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins that AnalyzeNode publishes the analyzer AnalysisResult into state.
    """

    async def test_analysis_result_is_published_to_state_from_plan_metadata(self) -> None:
        """
        state[ANALYSIS] must carry the AnalysisResult so downstream nodes read distinct reasoning.
        """

        analysis = AnalysisResult(
            screen_description="home screen with search bar and app icons",
            action=Action(action_type=ActionType.TAP, rationale="tap search"),
            reasoning="deliberating over the visible search bar and the goal",
        )
        agent_state = AgentState(
            intent="search for burgers",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

        planner = Mock()
        planner.plan_step = AsyncMock(
            return_value=SimpleNamespace(
                plan=SimpleNamespace(
                    metrics=None,
                    step=Step(
                        action=Action(
                            target="search_field",
                            rationale="tap search",
                            action_type=ActionType.TAP,
                        ),
                        metadata={},
                        step_number=0,
                        screen_hash="abc",
                    ),
                    reason="planned",
                    is_complete=False,
                    should_retry=False,
                    reasoning="deliberating",
                    context=PlanContext(analysis=analysis),
                ),
                events=(),
            ),
        )
        provider = SimpleNamespace(
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            is_cancelled=AsyncMock(return_value=False),
            context=SimpleNamespace(
                intent="test-intent",
                phase=AsyncMock(),
                max_steps=10,
                use_xml=True,
                reasoner=Mock(),
                planner=planner,
                workflow_id="run-test",
                agent_state=agent_state,
                memory=SimpleNamespace(set=AsyncMock()),
                context_manager=AgentFixtures.context_manager(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                clock=RunClock(),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(100, 200))),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
            ),
        )
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=100,
                    height=200,
                    activity="app",
                    image=b"png",
                    timestamp=1,
                )
            }
        )

        published = result[CommonStateKey.ANALYSIS]
        self.assertIsInstance(published, AnalysisResult)
        self.assertEqual(
            "deliberating over the visible search bar and the goal",
            published.reasoning,
        )
        self.assertEqual(
            "home screen with search bar and app icons",
            published.screen_description,
        )

    async def test_missing_analysis_metadata_leaves_state_analysis_none(self) -> None:
        """
        When planner does not attach ANALYSIS metadata, state[ANALYSIS] must be None (not crash).
        """

        agent_state = AgentState(
            intent="open the app",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        planner = Mock()
        planner.plan_step = AsyncMock(
            return_value=SimpleNamespace(
                plan=SimpleNamespace(
                    metrics=None,
                    step=None,
                    is_complete=False,
                    should_retry=False,
                    reason="",
                    reasoning="",
                    context=PlanContext(),
                ),
                events=(),
            ),
        )
        provider = SimpleNamespace(
            is_cancelled=AsyncMock(return_value=False),
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            context=SimpleNamespace(
                intent="test-intent",
                phase=AsyncMock(),
                workflow_id="run-test",
                max_steps=10,
                agent_state=agent_state,
                memory=SimpleNamespace(set=AsyncMock()),
                context_manager=AgentFixtures.context_manager(),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(100, 200))),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
                planner=planner,
                use_xml=True,
                reasoner=Mock(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                clock=RunClock(),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
            ),
        )
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        result = await node.run(
            state={
                CommonStateKey.CAPTURE: ScreenCapture(
                    width=100,
                    height=200,
                    activity="app",
                    image=b"png",
                    timestamp=1,
                )
            }
        )

        self.assertIsNone(result[CommonStateKey.ANALYSIS])
