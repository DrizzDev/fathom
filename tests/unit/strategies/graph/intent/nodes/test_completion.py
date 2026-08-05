from __future__ import annotations

import unittest
from typing import List, Optional
from unittest.mock import MagicMock

from fathom.constants import ActionType
from fathom.constants.observation import KeyboardVisibility
from fathom.constants.state import CommonStateKey, IntentStateKey, VerifyMode
from fathom.constants.turn.advancement import AdvanceKind
from fathom.core.agent.state import AgentState
from fathom.core.services.criterion import CriterionObserver
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.capture import Capture, CaptureRequest
from fathom.schemas.criterion import CriterionDecision, CriterionSource, CriterionVerdict
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.requirement import CommandRequirement, PressRequirement
from fathom.schemas.results import AnalysisResult, PlanContext, PlanResult
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.success import Success
from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator
from tests.builders import SubGoalFixtures, SuccessFixtures


class _StubCriterionChecker(CriterionObserver):
    """
    Deterministic criterion observer returning a fixed decision and counting calls.
    """

    def __init__(self, *, decision: CriterionDecision) -> None:
        self.__decision = decision
        self.calls: int = 0

    async def check(
        self,
        *,
        workflow_id: str,
        index: int,
        requirement: object,
        observation: ScreenObservation,
    ) -> CriterionDecision:
        """
        Return the configured decision, counting each invocation.
        """

        _ = (workflow_id, index, requirement, observation)
        self.calls += 1
        return self.__decision


class SubGoalEvaluatorTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins SubGoalEvaluator against canonical Success: pre/post-dispatch evidence and advancement.
    """

    __LOGIN = "tap the Login button"

    def __agent_state(self, goals: List[SubGoal]) -> AgentState:
        """
        Build a real AgentState seeded with the given canonical sub-goals.
        """

        state = AgentState(
            intent="test intent",
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        state.set_sub_goals(goals)
        return state

    @staticmethod
    def __context(state: AgentState) -> MagicMock:
        """
        Wrap a real AgentState in a graph-context surface.
        """

        context = MagicMock(name="GraphContext")
        context.agent_state = state
        context.workflow_id = "wf"
        return context

    def __evaluator(
        self, *, state: AgentState, verdict: Optional[CriterionVerdict]
    ) -> SubGoalEvaluator:
        """
        Build an evaluator whose observer returns the given verdict (or SATISFIED by default).
        """

        decision = CriterionDecision(
            verdict=verdict if verdict is not None else CriterionVerdict.SATISFIED,
            source=CriterionSource.SYMBOLIC,
            confidence=0.95,
            evidence=(),
            notes=None,
        )
        return SubGoalEvaluator(
            context=self.__context(state), criterion_observer=_StubCriterionChecker(decision=decision)
        )

    @staticmethod
    def __plan() -> PlanResult:
        """
        Build a minimal PlanResult carrying a synthetic AnalysisResult.
        """

        analysis = AnalysisResult(
            action=Action(action_type=ActionType.TAP, target="t", rationale="r", confidence=1.0),
            reasoning="r",
            screen_description="s",
            metadata={"tool_args": {}},
        )
        return PlanResult(
            step=None,
            is_complete=False,
            reason="t",
            context=PlanContext(analysis=analysis),
        )

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Build a minimal ScreenObservation for evaluator input.
        """

        element = PerceivedElement(
            identifier="e0",
            bounds=Bounds(x=0, y=0, width=10, height=10),
            source=ElementSource.XML,
            role=ElementRole.TEXT,
            confidence=1.0,
            text="dummy",
            tappable=False,
        )
        return ScreenObservation(
            activity="com.test.app",
            elements=(element,),
            hashes=ScreenHashBundle(
                visual_hash="vh0",
                xml_hash="0000000000000000",
                interaction_hash="0000000000000000",
            ),
            overlays=(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            scroll=(),
            calls_to_action=(),
            focused=None,
        )

    @staticmethod
    def __tap(*, target: str) -> Action:
        """
        Build a TAP action grounded on the given human-facing target.
        """

        return Action(
            action_type=ActionType.TAP,
            target="element",
            natural_language_target=target,
            rationale="r",
            confidence=1.0,
        )

    @classmethod
    def __result(
        cls,
        *,
        action: Optional[Action] = None,
        requirement: Optional[CommandRequirement] = None,
        executed: bool = True,
        capture: Optional[Capture] = None,
    ) -> StepResult:
        """
        Build a StepResult carrying an optional admitted requirement and capture.
        """

        step = Step(
            action=action if action is not None else cls.__tap(target="Login"),
            step_number=0,
            screen_hash="pre",
            requirement=requirement,
        )
        return StepResult(
            step=step,
            success=True,
            executed=executed,
            capture=capture,
            pre_hash="pre",
            post_hash="post",
            screen_changed=True,
            duration=1,
        )

    @classmethod
    def __store_result(cls, *, capture_name: str, executed: bool = True) -> StepResult:
        """
        Build a STORE StepResult whose committed capture uses the given name.
        """

        action = Action(
            action_type=ActionType.STORE,
            target="element",
            rationale="capture",
            capture=CaptureRequest(name="price", subject="item price", value="9.99"),
        )
        step = Step(action=action, step_number=0, screen_hash="pre")
        return StepResult(
            step=step,
            success=True,
            executed=executed,
            capture=Capture(name=capture_name, step=0, success=True, value="9.99"),
            pre_hash="pre",
            post_hash="pre",
            screen_changed=False,
            duration=1,
        )

    def __two_goals(self, first: Success) -> List[SubGoal]:
        """
        Build a two-goal plan so an advancing first goal loops back to GROUND (not final VERIFY).
        """

        return [
            SubGoalFixtures.make(index=0, description="first", success=first),
            SubGoalFixtures.make(index=1, description="second"),
        ]

    # ── Observed success ──────────────────────────────────────────────────

    async def test_observed_goal_advances_on_satisfied_verdict_post_dispatch(self) -> None:
        """
        An observed goal advances when its own observation is freshly satisfied after dispatch.
        """

        state = self.__agent_state(self.__two_goals(SuccessFixtures.observed(assertion="home")))
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.SATISFIED)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__result(),
            accumulated=[],
            observation=self.__observation(),
        )

        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(state.current_sub_goal_index, 1)

    async def test_pre_dispatch_observed_satisfaction_advances_via_probe(self) -> None:
        """
        An observed goal already satisfied on the settled screen advances before any dispatch.
        """

        state = self.__agent_state(self.__two_goals(SuccessFixtures.observed(assertion="home")))
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.SATISFIED)

        result = await evaluator.probe(observation=self.__observation())

        self.assertIs(result.advancement.kind, AdvanceKind.SATISFIED_PRIOR)
        assert result.transition is not None
        self.assertTrue(result.transition.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(state.current_sub_goal_index, 1)

    async def test_observed_goal_retains_when_verdict_unsatisfied(self) -> None:
        """
        An observed goal retains when its observation is not satisfied.
        """

        state = self.__agent_state([SubGoalFixtures.make(success=SuccessFixtures.observed())])
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.UNSATISFIED)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__result(),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    # ── Command success ───────────────────────────────────────────────────

    def __command(self, *, postcondition_assertion: Optional[str] = None) -> Success:
        return SuccessFixtures.command(
            requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            postcondition=(
                SuccessFixtures.observation(postcondition_assertion)
                if postcondition_assertion is not None
                else None
            ),
            quote="tap",
            intent=self.__LOGIN,
        )

    async def test_command_goal_never_advances_pre_dispatch(self) -> None:
        """
        A command goal is never advanced before its command executes, even when probed.
        """

        state = self.__agent_state([SubGoalFixtures.make(success=self.__command())])
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.SATISFIED)

        result = await evaluator.probe(observation=self.__observation())

        self.assertIsNone(result.transition)
        self.assertIs(result.advancement.kind, AdvanceKind.RETAIN)

    async def test_command_goal_retains_on_preparatory_action(self) -> None:
        """
        A preparatory action carrying no admitted requirement cannot advance a command goal.
        """

        success = self.__command()
        state = self.__agent_state([SubGoalFixtures.make(success=success)])
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.SATISFIED)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__result(action=self.__tap(target="Login"), requirement=None),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_command_goal_advances_on_matching_execution(self) -> None:
        """
        A command goal advances when the admitted requirement matches the executed action.
        """

        success = self.__command()
        state = self.__agent_state(self.__two_goals(success))
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.SATISFIED)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__result(
                action=self.__tap(target="Login"),
                requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            ),
            accumulated=[],
            observation=self.__observation(),
        )

        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))

    async def test_command_goal_with_wrong_postcondition_retains(self) -> None:
        """
        A command with a postcondition retains when the postcondition verdict is unsatisfied.
        """

        success = self.__command(postcondition_assertion="home shown")
        state = self.__agent_state([SubGoalFixtures.make(success=success)])
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.UNSATISFIED)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__result(
                action=self.__tap(target="Login"),
                requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            ),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    # ── Capture success ───────────────────────────────────────────────────

    async def test_capture_goal_advances_on_matching_store(self) -> None:
        """
        A capture goal advances exactly once on a committed STORE matching the requested identity.
        """

        success = SuccessFixtures.capture(name="price", subject="item price")
        state = self.__agent_state(self.__two_goals(success))
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.UNCLEAR)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__store_result(capture_name="price"),
            accumulated=[],
            observation=self.__observation(),
        )

        assert result is not None
        self.assertTrue(result.get(IntentStateKey.SHOULD_RETRY))
        self.assertEqual(state.current_sub_goal_index, 1)

    async def test_capture_identity_mismatch_retains(self) -> None:
        """
        A committed capture under a different name cannot advance the capture goal.
        """

        success = SuccessFixtures.capture(name="price", subject="item price")
        state = self.__agent_state([SubGoalFixtures.make(success=success)])
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.UNCLEAR)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__store_result(capture_name="other"),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    async def test_capture_goal_retains_when_store_not_executed(self) -> None:
        """
        A successful-looking capture with executed=False cannot advance the capture goal.
        """

        success = SuccessFixtures.capture(name="price", subject="item price")
        state = self.__agent_state([SubGoalFixtures.make(success=success)])
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.UNCLEAR)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__store_result(capture_name="price", executed=False),
            accumulated=[],
            observation=self.__observation(),
        )

        self.assertIsNone(result)

    # ── Final goal handoff ────────────────────────────────────────────────

    async def test_final_goal_routes_to_verify_without_commit(self) -> None:
        """
        Advancing the terminal goal routes to VERIFY for final adjudication rather than completing here.
        """

        state = self.__agent_state([SubGoalFixtures.make(success=SuccessFixtures.observed())])
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.SATISFIED)

        result = await evaluator.evaluate(
            plan=self.__plan(),
            step_result=self.__result(),
            accumulated=[],
            observation=self.__observation(),
        )

        assert result is not None
        self.assertEqual(
            result.get(IntentStateKey.VERIFY_MODE), VerifyMode.PENDING_FINAL_COMMIT.value
        )
        self.assertTrue(result.get(CommonStateKey.IS_COMPLETE))


if __name__ == "__main__":
    unittest.main()
