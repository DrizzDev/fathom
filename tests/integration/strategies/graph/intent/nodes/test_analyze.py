from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, Mock

from tests.builders import ActionFixtures, ScreenFixtures, SubGoalFixtures, SuccessFixtures
from tests.builders.agent import AgentFixtures

from fathom.constants import ActionType
from fathom.constants.assessment import VisualVerdict
from fathom.constants.observation import KeyboardVisibility as _KeyboardVisibility
from fathom.constants.retries import RetryBranch, RetryKind
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.constants.turn.advancement import AdvanceKind
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.core.capability.catalog import CommandCatalogProvider as _CommandCatalogProvider
from fathom.core.services.criterion import CriterionObserver as _CriterionObserverBase
from fathom.schemas.actions import Action
from fathom.schemas.actions import Bounds as _Bounds
from fathom.schemas.advancement import Advancement
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.criterion import CriterionDecision as _CriterionDecision
from fathom.schemas.criterion import CriterionSource as _CriterionSource
from fathom.schemas.criterion import CriterionVerdict as _CriterionVerdict
from fathom.schemas.observation import ElementRole as _ElementRole
from fathom.schemas.observation import ElementSource as _ElementSource
from fathom.schemas.observation import KeyboardObservation as _KeyboardObservation
from fathom.schemas.observation import PerceivedElement as _PerceivedElement
from fathom.schemas.observation import ScreenObservation as _ScreenObservation
from fathom.schemas.planner import PlannerMetrics
from fathom.schemas.results import AnalysisResult, PlanContext, PlanResult, PlanTurn
from fathom.schemas.retries import RetryLimits
from fathom.schemas.screens import ScreenHashBundle as _ScreenHashBundle
from fathom.strategies.graph.intent.nodes.analyze import AnalyzeNode
from fathom.strategies.graph.intent.nodes.completion import ProbeResult
from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator as _SubGoalEvaluator
from fathom.strategies.graph.state import IntentGraphState

RETAIN_PROBE = ProbeResult(advancement=Advancement(kind=AdvanceKind.RETAIN))


class _Persistence:
    """
    Captures the last persisted IntentGraphState so the test can assert post-termination payload shape.
    """

    def __init__(self) -> None:
        self.last: IntentGraphState = {}

    def restore(self, *, state: IntentGraphState) -> None:
        """
        Restore is a no-op; AgentState already lives in-memory across iterations.
        """

        _ = state

    def persist(self, *, result: IntentGraphState) -> None:
        """
        Snapshot the persisted payload for assertions.
        """

        self.last = result


class _RealPlannerHarness:
    """
    Drives ``AnalyzeNode`` through the production ``StepPlanner`` with only the LLM boundary (``VisionService.analyze``) stubbed.
    """

    BLOCKED_DESCRIPTOR: str = "Swipe left on More on Swiggy widget"

    def __init__(self, *, agent_state: AgentState) -> None:
        self.__agent_state = agent_state

        self.__action = self.__build_blocked_action()
        self.__vision = self.__build_vision_stub(action=self.__action)
        self.__reasoner = self.__build_reasoner_stub(action=self.__action)

        self.__persistence = _Persistence()
        self.__planner = StepPlanner(vision_tool=self.__vision)

    @property
    def persistence(self) -> _Persistence:
        """
        Expose the persistence sink for assertions.
        """

        return self.__persistence

    def seed_blocked_action(self) -> None:
        """
        Mark the action as already-failed so the real planner's ``should_avoid_action`` branch fires on the next call.
        """

        self.__agent_state.record_repeated_action_failure(action=self.__action)

    def provider(self) -> SimpleNamespace:
        """
        Build the ``IntentNodeProvider``-shaped namespace consumed by :class:`AnalyzeNode`.
        """

        return SimpleNamespace(
            persistence=self.__persistence,
            hitl=SimpleNamespace(prompt=AsyncMock()),
            is_cancelled=AsyncMock(return_value=False),
            completion=SimpleNamespace(probe=AsyncMock(return_value=RETAIN_PROBE)),
            context=SimpleNamespace(
                max_steps=10,
                use_xml=True,
                workflow_id="run-test",
                planner=self.__planner,
                reasoner=self.__reasoner,
                agent_state=self.__agent_state,
                context_manager=AgentFixtures.context_manager(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                memory=SimpleNamespace(set=AsyncMock()),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(1080, 2340))),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
            ),
        )

    @classmethod
    def __build_blocked_action(cls) -> Action:
        """
        Construct the action the LLM keeps proposing on the unchanged screen.
        """

        return ActionFixtures.make(
            target=cls.BLOCKED_DESCRIPTOR,
            action_type=ActionType.SWIPE_LEFT,
            natural_language_target=cls.BLOCKED_DESCRIPTOR,
            rationale="Scroll the More widget to reveal Gourmet delights.",
        )

    @staticmethod
    def __build_vision_stub(*, action: Action) -> Mock:
        """
        Stub ``VisionService.analyze`` to return the blocked action and a benign rejection-history payload.
        """

        analysis = AnalysisResult(
            action=action,
            metadata={"tool_args": {}},
            screen_description="Home feed",
            reasoning="The carousel needs to scroll left.",
        )
        rejection_history: List[ConversationTurn] = []

        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=rejection_history)

        return vision

    @staticmethod
    def __build_reasoner_stub(*, action: Action) -> Mock:
        """
        Stub the reasoner to pick the analysis-supplied action so the planner reaches the avoidance check.
        """

        reasoner = Mock()
        reasoner.select_best_action.return_value = action

        return reasoner


class AnalyzeNodePlannerRetryBudgetIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """
    Regression: the planner rejecting the same action on an unchanged screen used to loop forever;
    the retry budget must terminate the workflow at ``cap`` iterations.
    """

    @staticmethod
    def __state(*, planner_cap: int) -> AgentState:
        """
        Build a non-interactive agent state with the requested planner retry cap.
        """

        return AgentState(
            max_steps=10,
            intent="Find Gourmet delights",
            retries=RetryLimits(planner=planner_cap),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )

    @staticmethod
    def __capture() -> object:
        """
        Build a constant capture so every iteration runs against an unchanged screen.
        """

        return ScreenFixtures.capture(activity="bundl.swiggy", width=1080, height=2340)

    async def test_silent_rejection_loop_terminates_at_budget(self) -> None:
        """
        With the real ``StepPlanner`` firing ``should_avoid_action`` on every turn, the workflow must terminate at ``cap`` with ``RETRY_BUDGET_EXHAUSTED``.
        """

        agent_state = self.__state(planner_cap=5)
        harness = _RealPlannerHarness(agent_state=agent_state)

        harness.seed_blocked_action()
        node = AnalyzeNode(provider=harness.provider())  # type: ignore[arg-type]

        terminal: IntentGraphState = {}
        for index in range(agent_state.retries.planner.cap + 2):
            terminal = await node.run(state={CommonStateKey.CAPTURE: self.__capture()})
            if terminal.get(CommonStateKey.IS_COMPLETE):
                break

            self.assertEqual(
                agent_state.retries.planner.count,
                index + 1,
                msg=f"iteration {index}: budget not advancing",
            )

        self.assertTrue(terminal.get(CommonStateKey.IS_COMPLETE))
        self.assertFalse(terminal.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(
            terminal.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.RETRY_BUDGET_EXHAUSTED.value,
        )
        self.assertEqual(agent_state.step_count, 0)
        self.assertEqual(agent_state.retries.planner.count, agent_state.retries.planner.cap)

    async def test_terminal_retry_exhaustion_bypasses_complete_deferral(self) -> None:
        """
        When the planner returns a terminal ``RETRY_BUDGET_EXHAUSTED`` verdict,
        the complete-deferral gate must not swallow it — even with sub-goals still open the very first iteration must terminate.
        """

        agent_state = self.__state(planner_cap=2)
        agent_state.set_sub_goals(
            [
                SubGoalFixtures.make(description="Scroll left to find Gourmet"),
                SubGoalFixtures.make(description="Tap the Gourmet tile"),
            ]
        )
        for _ in range(agent_state.retries.planner.cap):
            agent_state.tick_planner_retry(
                kind=RetryKind.SILENT_REJECTION,
                branch=RetryBranch.SHOULD_AVOID_ACTION,
                action=_RealPlannerHarness.BLOCKED_DESCRIPTOR,
            )
        self.assertTrue(agent_state.retries.planner.exhausted)

        harness = _RealPlannerHarness(agent_state=agent_state)
        node = AnalyzeNode(provider=harness.provider())  # type: ignore[arg-type]

        terminal = await node.run(state={CommonStateKey.CAPTURE: self.__capture()})

        self.assertTrue(terminal.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            terminal.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.RETRY_BUDGET_EXHAUSTED.value,
        )
        self.assertEqual(agent_state.consecutive_complete_deferrals, 0)
        self.assertFalse(agent_state.all_sub_goals_complete())

    async def test_planner_retry_count_survives_checkpoint_restore_mid_loop(self) -> None:
        """
        Production runs persist AgentState mid-loop on every node transition;
        the retry counter must survive the round-trip or a restore would reset progress toward the cap and let the loop run forever.
        """

        live = self.__state(planner_cap=5)
        harness = _RealPlannerHarness(agent_state=live)
        harness.seed_blocked_action()
        node = AnalyzeNode(provider=harness.provider())  # type: ignore[arg-type]

        for _ in range(3):
            await node.run(state={CommonStateKey.CAPTURE: self.__capture()})

        self.assertEqual(live.retries.planner.count, 3)

        restored = AgentState.from_checkpoint(
            live.to_checkpoint(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        self.assertEqual(restored.retries.planner.count, 3)

        restored_harness = _RealPlannerHarness(agent_state=restored)
        restored.record_repeated_action_failure(
            action=ActionFixtures.make(
                rationale="Restored",
                action_type=ActionType.SWIPE_LEFT,
                target=_RealPlannerHarness.BLOCKED_DESCRIPTOR,
                natural_language_target=_RealPlannerHarness.BLOCKED_DESCRIPTOR,
            )
        )
        restored_node = AnalyzeNode(provider=restored_harness.provider())  # type: ignore[arg-type]

        terminal = await restored_node.run(state={CommonStateKey.CAPTURE: self.__capture()})
        self.assertFalse(terminal.get(CommonStateKey.IS_COMPLETE))

        terminal = await restored_node.run(state={CommonStateKey.CAPTURE: self.__capture()})
        self.assertTrue(terminal.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            terminal.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.RETRY_BUDGET_EXHAUSTED.value,
        )

    async def test_stuck_verdict_with_open_sub_goals_is_not_deferred(self) -> None:
        """
        STUCK with open sub-goals must not be coerced back into a deferral loop in AnalyzeNode; downstream routing (VERIFY vs END for STUCK) is governed by ``TERMINAL_COMPLETION_REASONS`` and is intentionally out of scope here.
        """

        agent_state = self.__state(planner_cap=2)
        agent_state.set_sub_goals(
            [
                SubGoalFixtures.make(description="Scroll left to find Gourmet"),
                SubGoalFixtures.make(description="Tap the Gourmet tile"),
            ]
        )
        agent_state.mark_complete(reason=CompletionReason.STUCK.value)

        harness = _RealPlannerHarness(agent_state=agent_state)
        node = AnalyzeNode(provider=harness.provider())  # type: ignore[arg-type]

        terminal = await node.run(state={CommonStateKey.CAPTURE: self.__capture()})

        self.assertTrue(terminal.get(CommonStateKey.IS_COMPLETE))
        self.assertEqual(
            terminal.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.STUCK.value,
        )
        self.assertEqual(agent_state.consecutive_complete_deferrals, 0)

    async def test_no_capture_terminates_immediately(self) -> None:
        """
        Device adapter already retries grounding internally; an analyze-level capture failure must terminate fast, not eat the planner budget.
        """

        agent_state = self.__state(planner_cap=5)

        harness = _RealPlannerHarness(agent_state=agent_state)
        node = AnalyzeNode(provider=harness.provider())  # type: ignore[arg-type]

        terminal = await node.run(state={CommonStateKey.CAPTURE: None})

        self.assertTrue(terminal.get(CommonStateKey.IS_COMPLETE))
        self.assertFalse(terminal.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(
            terminal.get(CommonStateKey.COMPLETION_REASON),
            CompletionReason.FAILED.value,
        )
        self.assertEqual(agent_state.retries.planner.count, 0)


if __name__ == "__main__":
    unittest.main()


class _SatisfiedCriterionObserver(_CriterionObserverBase):
    """
    Criterion observer stub that always reports high-confidence SATISFIED on the settled screen.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def check(
        self, *, workflow_id: str, index: int, requirement: object, observation: _ScreenObservation
    ) -> _CriterionDecision:
        """
        Return a fresh high-confidence SATISFIED reading regardless of inputs.
        """

        _ = (workflow_id, index, requirement, observation)
        self.calls += 1
        return _CriterionDecision(
            verdict=_CriterionVerdict.SATISFIED,
            source=_CriterionSource.SYMBOLIC,
            confidence=0.9,
            evidence=("rating >= 4.2 visible",),
        )


class AnalyzeNodePreDispatchSatisfiedPriorTest(unittest.IsolatedAsyncioTestCase):
    """
    Graph-level: an active goal already satisfied on the settled screen advances via SATISFIED_PRIOR
    inside ANALYZE after the single planner call, discarding the planned action before EXECUTE. The real
    CriterionObserver-shaped stub, AdvancementPolicy and SubGoalEvaluator are exercised through the
    production AnalyzeNode; only the criterion reading is fixed.
    """

    @staticmethod
    def __observation() -> _ScreenObservation:
        """
        Build a minimal settled-screen observation.
        """

        element = _PerceivedElement(
            identifier="e0",
            bounds=_Bounds(x=0, y=0, width=10, height=10),
            source=_ElementSource.XML,
            role=_ElementRole.TEXT,
            confidence=1.0,
            text="Ghar Magic Soap 4.9",
            tappable=False,
        )
        return _ScreenObservation(
            activity="com.meesho.supply",
            elements=(element,),
            hashes=_ScreenHashBundle(
                visual_hash="vh0",
                xml_hash="0000000000000000",
                interaction_hash="0000000000000000",
            ),
            overlays=(),
            keyboard=_KeyboardObservation(visibility=_KeyboardVisibility.HIDDEN),
            scroll=(),
            calls_to_action=(),
            focused=None,
        )

    async def test_satisfied_goal_advances_before_dispatch_without_execute(self) -> None:
        """
        The already-satisfied goal advances to the next sub-goal after the single planner call, before EXECUTE.
        """

        agent_state = AgentState(
            intent="buy ghar soap",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.set_sub_goals(
            [
                SubGoalFixtures.make(
                    index=0,
                    description="Scroll until rating >= 4.2",
                    success=SuccessFixtures.observed(
                        assertion="a product with rating >= 4.2 is visible"
                    ),
                ),
                SubGoalFixtures.make(
                    index=1,
                    description="Select the product",
                    success=SuccessFixtures.observed(assertion="product detail open"),
                ),
            ]
        )
        catalog = _CommandCatalogProvider().build()
        checker = _SatisfiedCriterionObserver()
        evaluator = _SubGoalEvaluator(
            context=SimpleNamespace(
                agent_state=agent_state, workflow_id="run-test", catalog=catalog
            ),
            criterion_observer=checker,
        )
        planner = Mock()
        planner.plan_step = AsyncMock(
            return_value=PlanTurn(
                plan=PlanResult(
                    reason="already satisfied",
                    is_complete=False,
                    step=None,
                    context=PlanContext(
                        analysis=AnalysisResult(reasoning="r", screen_description="s")
                    ),
                )
            )
        )
        provider = SimpleNamespace(
            is_cancelled=AsyncMock(return_value=False),
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            completion=evaluator,
            context=SimpleNamespace(
                workflow_id="run-test",
                max_steps=10,
                agent_state=agent_state,
                context_manager=AgentFixtures.context_manager(),
                planner=planner,
                use_xml=True,
                reasoner=Mock(),
                memory=SimpleNamespace(set=AsyncMock()),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
                catalog=catalog,
            ),
        )
        state: IntentGraphState = {
            CommonStateKey.CAPTURE: ScreenFixtures.capture(activity="com.meesho.supply"),
            CommonStateKey.SCREEN_OBSERVATION: self.__observation(),
        }

        result = await AnalyzeNode(provider=provider).run(state=state)

        self.assertEqual(checker.calls, 1)
        planner.plan_step.assert_awaited_once()
        self.assertEqual(agent_state.current_sub_goal_index, 1)
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertIsNone(result.get(IntentStateKey.PLANNED_STEP))


class AnalyzeNodeShadowAssessmentIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """
    Graph-level proof of Slice-2 shadow recording: the real AnalyzeNode + production StepPlanner,
    with the vision boundary stubbed, records a SATISFIED+action divergence while the live action
    still reaches the planned step and the goal cursor never advances.
    """

    def __provider(self, *, agent_state: AgentState, vision: Mock, reasoner: Mock) -> SimpleNamespace:
        return SimpleNamespace(
            persistence=_Persistence(),
            hitl=SimpleNamespace(prompt=AsyncMock()),
            is_cancelled=AsyncMock(return_value=False),
            completion=SimpleNamespace(probe=AsyncMock(return_value=RETAIN_PROBE)),
            context=SimpleNamespace(
                max_steps=10,
                use_xml=True,
                workflow_id="run-test",
                planner=StepPlanner(vision_tool=vision),
                reasoner=reasoner,
                agent_state=agent_state,
                context_manager=AgentFixtures.context_manager(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
                memory=SimpleNamespace(set=AsyncMock()),
                telemetry=SimpleNamespace(info=AsyncMock(), error=AsyncMock()),
                signal=SimpleNamespace(supports_interruption=Mock(return_value=False)),
                device=SimpleNamespace(get_dimensions=AsyncMock(return_value=(1080, 2340))),
                configuration=SimpleNamespace(intent=SimpleNamespace(prompt_user_if_stuck=False)),
            ),
        )

    async def test_satisfied_with_action_records_retaining_candidate_action_stays_cursor_holds(self) -> None:
        agent_state = AgentState(
            intent="Open Amazon",
            max_steps=10,
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        agent_state.set_sub_goals([SubGoalFixtures.make(description="Amazon home is shown")])
        cursor_before = agent_state.current_sub_goal_index

        action = ActionFixtures.make(
            target="Continue",
            action_type=ActionType.TAP,
            natural_language_target="Continue",
            rationale="tap continue",
        )
        analysis = AnalysisResult(
            action=action,
            metadata={"tool_args": {}},
            reasoning="home visible",
            screen_description="Amazon home",
            visual_assessment=VisualAssessment(
                verdict=VisualVerdict.SATISFIED, confidence=0.9, evidence="home visible"
            ),
            planner=PlannerMetrics(latency=0.5, calls=1),
        )
        vision = Mock()
        vision.analyze = AsyncMock(return_value=analysis)
        vision.build_rejection_history_from_analysis = Mock(return_value=[])
        reasoner = Mock()
        reasoner.select_best_action.return_value = action

        node = AnalyzeNode(provider=self.__provider(agent_state=agent_state, vision=vision, reasoner=reasoner))  # type: ignore[arg-type]

        with self.assertLogs(
            "fathom.strategies.graph.intent.nodes.analyze", level="INFO"
        ) as logs:
            result = await node.run(state={CommonStateKey.CAPTURE: ScreenFixtures.capture(activity="com.amazon.mp3")})

        # Exactly one model turn (the vision boundary the planner calls once).
        vision.analyze.assert_awaited_once()
        # The live action still reaches the planned step.
        self.assertIsNotNone(result[IntentStateKey.PLANNED_STEP])
        # A dispatched turn does not emit at Analyze; it carries the draft for CompletionNode to finalize.
        comparisons = [
            record.__dict__["shadow.record"]
            for record in logs.records
            if record.__dict__.get("event") == "shadow.turn.comparison"
        ]
        self.assertEqual(len(comparisons), 0)
        draft = result[IntentStateKey.PLAN].context.shadow
        self.assertIsNotNone(draft)
        # A SATISFIED verdict with an action retains as candidate on the carried draft.
        self.assertIs(draft.pre_dispatch.candidate.kind, AdvanceKind.RETAIN)
        # The goal cursor did not advance from the shadow path.
        self.assertEqual(agent_state.current_sub_goal_index, cursor_before)
