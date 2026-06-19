from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, Mock

from tests.builders import ActionFixtures, ScreenFixtures, SubGoalFixtures
from tests.builders.agent import AgentFixtures

from fathom.constants import ActionType
from fathom.constants.retries import RetryBranch, RetryKind
from fathom.constants.state import CommonStateKey, CompletionReason, IntentStateKey
from fathom.core.agent.planner import StepPlanner
from fathom.core.agent.state import AgentState
from fathom.schemas.actions import Action
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.conversation import ConversationTurn
from fathom.schemas.results import AnalysisResult
from fathom.schemas.retries import RetryLimits
from fathom.strategies.graph.intent.nodes.analyze import AnalyzeNode
from fathom.strategies.graph.state import IntentGraphState


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
            context=SimpleNamespace(
                max_steps=10,
                use_xml=True,
                workflow_id="run-test",
                planner=self.__planner,
                reasoner=self.__reasoner,
                agent_state=self.__agent_state,
                context_manager=AgentFixtures.context_manager(),
                metrics=SimpleNamespace(record=Mock(), record_tokens=Mock()),
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

    async def test_invalid_retry_metadata_emits_structured_warning(self) -> None:
        """
        Planner-stamped retry metadata is a producer contract; missing/unknown KIND or BRANCH must surface as a ``PLANNER_RETRY_METADATA_INVALID`` log so contract drift is visible in production before it silently consumes the wrong budget.
        """

        agent_state = self.__state(planner_cap=5)
        harness = _RealPlannerHarness(agent_state=agent_state)
        provider = harness.provider()

        plan_with_bogus_metadata = SimpleNamespace(
            step=None,
            memories=0,
            metrics=None,
            is_complete=False,
            should_retry=True,
            reason=CompletionReason.FAILED.value,
            metadata={"RETRY_KIND": "NOT_A_REAL_KIND", "RETRY_BRANCH": "NOT_A_REAL_BRANCH"},
        )
        provider.context.planner = SimpleNamespace(
            plan_step=AsyncMock(return_value=plan_with_bogus_metadata)
        )
        node = AnalyzeNode(provider=provider)  # type: ignore[arg-type]

        with self.assertLogs(
            "fathom.strategies.graph.intent.nodes.analyze", level="WARNING"
        ) as logs:
            await node.run(state={CommonStateKey.CAPTURE: self.__capture()})

        invalid_records = [
            record
            for record in logs.records
            if getattr(record, "event", None) == "PLANNER_RETRY_METADATA_INVALID"
        ]
        self.assertEqual(len(invalid_records), 2, msg="expected one warning per invalid field")

        flagged_raws = {record.__dict__["metadata.raw"] for record in invalid_records}
        flagged_fields = {record.__dict__["metadata.field"] for record in invalid_records}

        self.assertEqual(flagged_fields, {"RETRY_KIND", "RETRY_BRANCH"})
        self.assertEqual(flagged_raws, {"'NOT_A_REAL_KIND'", "'NOT_A_REAL_BRANCH'"})

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
