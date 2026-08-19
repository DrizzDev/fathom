from __future__ import annotations

import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.constants.assessment import PhaseComparison, VisualVerdict
from fathom.constants.observation import KeyboardVisibility
from fathom.constants.success import CaptureNameProvenance
from fathom.constants.turn.advancement import AdvanceKind
from fathom.core.agent.state import AgentState
from fathom.core.services.criterion import CriterionObserver
from fathom.core.services.outcome import OutcomeObserver
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.advancement import Advancement
from fathom.schemas.artifacts import ScreenArtifact, ScreenArtifactBundle, StepArtifacts
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.capabilities import HITLCapability, RuntimeCapabilities
from fathom.schemas.capture import Capture, CaptureIdentity, CaptureRequest
from fathom.schemas.criterion import CriterionDecision, CriterionSource, CriterionVerdict
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    PerceivedElement,
    ScreenObservation,
)
from fathom.schemas.planner import PlannerMetrics
from fathom.schemas.requirement import PressRequirement
from fathom.schemas.results import AnalysisResult, PlanContext, PlanResult
from fathom.schemas.screens import ScreenHashBundle
from fathom.schemas.shadow import GoalCursor, ShadowTurnDraft
from fathom.schemas.steps import Step, StepResult
from fathom.schemas.subgoal import GoalState, SubGoal
from fathom.schemas.success import CaptureSuccess, Success
from fathom.schemas.target import TargetAuthority
from fathom.strategies.graph.intent.nodes.completion import SubGoalEvaluator
from fathom.strategies.graph.intent.nodes.shadow import ShadowRunner
from tests.builders.success import SuccessFixtures

LOGGER = "fathom.strategies.graph.intent.nodes.completion"


class _StubChecker(CriterionObserver):
    """
    Deterministic criterion observer returning a fixed decision.
    """

    def __init__(self, *, verdict: CriterionVerdict) -> None:
        self.__verdict = verdict

    async def check(
        self, *, workflow_id: str, index: int, requirement: object, observation: ScreenObservation
    ) -> CriterionDecision:
        """
        Return the configured decision.
        """

        _ = (workflow_id, index, requirement, observation)
        return CriterionDecision(
            verdict=self.__verdict, source=CriterionSource.SYMBOLIC, confidence=0.95, evidence=()
        )


class _StubOutcome(OutcomeObserver):
    """
    Deterministic post-action vision observer returning a fixed assessment without an LLM call.
    """

    def __init__(self, *, verdict: VisualVerdict) -> None:
        self.__verdict = verdict

    async def assess(self, *, requirement: object, before: bytes, after: bytes) -> VisualAssessment:
        """
        Return the configured assessment.
        """

        _ = (requirement, before, after)
        return VisualAssessment(verdict=self.__verdict, confidence=0.9, evidence="stub")


class CompletionShadowFinalizationTest(unittest.IsolatedAsyncioTestCase):
    """
    CompletionNode finalizes the pre-dispatch draft into a receipt-bearing record and reconciles pending proof.
    """

    @staticmethod
    def __state(*, success: Success) -> AgentState:
        """
        Build an agent state with the active goal followed by a filler goal.
        """

        state = AgentState(
            intent="run", capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False))
        )
        state.set_sub_goals(
            [
                SubGoal(index=0, objective="active", success=success),
                SubGoal(index=1, objective="next", success=SuccessFixtures.observed()),
            ]
        )
        return state

    @staticmethod
    def __evaluator(
        *,
        state: AgentState,
        verdict: CriterionVerdict,
        outcome: Optional[OutcomeObserver] = None,
    ) -> SubGoalEvaluator:
        """
        Build the real evaluator over a graph-context surface wrapping the agent state.
        """

        context = MagicMock(name="GraphContext")
        context.agent_state = state
        context.workflow_id = "wf"
        context.phase = AsyncMock()
        return SubGoalEvaluator(
            context=context, criterion_observer=_StubChecker(verdict=verdict), outcome=outcome
        )

    @staticmethod
    def __observed_receipt(*, artifacts: bool = True) -> StepResult:
        """
        Build an executed TAP receipt for an observed goal, optionally carrying before/after screen bytes.
        """

        action = Action(action_type=ActionType.TAP, natural_language_target="x", rationale="r")
        step = Step(action=action, screen_hash="h", step_number=1)
        bundle = (
            StepArtifacts(
                screen=ScreenArtifactBundle(
                    before=ScreenArtifact(image=b"before"),
                    after=ScreenArtifact(image=b"after"),
                )
            )
            if artifacts
            else None
        )
        return StepResult(
            step=step,
            success=True,
            executed=True,
            pre_hash="a",
            post_hash="b",
            screen_changed=True,
            duration=5,
            artifacts=bundle,
        )

    @staticmethod
    def __analysis() -> AnalysisResult:
        """
        Build a post-dispatch analysis carrying producer metrics.
        """

        return AnalysisResult(
            reasoning="r",
            screen_description="s",
            planner=PlannerMetrics(latency=1.5, calls=1),
        )

    def __draft(self, *, active: GoalState) -> ShadowTurnDraft:
        """
        Build the pre-dispatch draft Analyze would have produced for the active goal.
        """

        return ShadowRunner().draft(
            workflow_id="wf",
            active=active,
            analysis=self.__analysis(),
            metrics=PlannerMetrics(latency=1.5, calls=1),
            screen="pre-dispatch-hash",
            foreground="app",
            authority=TargetAuthority.unbound(),
            live_pre=Advancement(kind=AdvanceKind.RETAIN),
            cursor_before=GoalCursor(index=0, total=2),
        )

    def __plan(self, *, draft: ShadowTurnDraft) -> PlanResult:
        """
        Build a dispatched plan carrying the pre-dispatch draft for finalization.
        """

        return PlanResult(
            reason="dispatched",
            is_complete=False,
            step=None,
            context=PlanContext(analysis=self.__analysis(), shadow=draft),
        )

    @staticmethod
    def __command_receipt(
        *, target: str = "Login", executed: bool = True, success: bool = True
    ) -> StepResult:
        """
        Build a matching TAP command receipt.
        """

        requirement = PressRequirement(operation=ActionType.TAP, target=target)
        action = Action(action_type=ActionType.TAP, natural_language_target=target, rationale="r")
        step = Step(action=action, screen_hash="h", step_number=1, requirement=requirement)
        return StepResult(
            step=step,
            success=success,
            executed=executed,
            pre_hash="a",
            post_hash="b",
            screen_changed=True,
            duration=7,
        )

    @staticmethod
    def __capture_receipt() -> StepResult:
        """
        Build a committed STORE receipt for the canonical capture name.
        """

        action = Action(
            action_type=ActionType.STORE,
            target="t",
            rationale="r",
            capture=CaptureRequest(name="price", subject="price", value="4.99"),
        )
        step = Step(action=action, screen_hash="h", step_number=1)
        return StepResult(
            step=step,
            success=True,
            executed=True,
            capture=Capture(name="price", step=1, success=True, value="4.99"),
            pre_hash="a",
            post_hash="b",
            screen_changed=True,
            duration=3,
        )

    @staticmethod
    def __command(*, postcondition: Optional[str] = None) -> Success:
        """
        Build a command success, optionally with a visual postcondition.
        """

        return SuccessFixtures.command(
            requirement=PressRequirement(operation=ActionType.TAP, target="Login"),
            postcondition=SuccessFixtures.observation(postcondition) if postcondition else None,
        )

    @staticmethod
    def __observation() -> ScreenObservation:
        """
        Build a minimal post-dispatch settled-screen observation.
        """

        element = PerceivedElement(
            identifier="e0",
            bounds=Bounds(x=0, y=0, width=10, height=10),
            source=ElementSource.XML,
            role=ElementRole.TEXT,
            confidence=1.0,
            text="x",
            tappable=False,
        )
        return ScreenObservation(
            activity="app",
            elements=(element,),
            hashes=ScreenHashBundle(
                visual_hash="post-dispatch-hash",
                xml_hash="0000000000000000",
                interaction_hash="0000000000000000",
            ),
            overlays=(),
            keyboard=KeyboardObservation(visibility=KeyboardVisibility.HIDDEN),
            scroll=(),
            calls_to_action=(),
            focused=None,
        )

    async def __records(
        self,
        *,
        evaluator: SubGoalEvaluator,
        plan: PlanResult,
        receipt: StepResult,
        observation: Optional[ScreenObservation] = None,
    ) -> List[Dict[str, object]]:
        """
        Run one completion turn and return every finalized shadow record it emitted.
        """

        with self.assertLogs(LOGGER, level="INFO") as logs:
            await evaluator.evaluate(
                plan=plan, step_result=receipt, accumulated=[], observation=observation
            )
        return [
            record.__dict__["shadow.record"]
            for record in logs.records
            if record.__dict__.get("event") == "shadow.turn.comparison"
        ]

    def __setup(
        self, *, success: Success, verdict: CriterionVerdict
    ) -> Tuple[SubGoalEvaluator, PlanResult, AgentState]:
        """
        Build the evaluator, the draft-carrying plan, and the state for one graph turn.
        """

        state = self.__state(success=success)
        evaluator = self.__evaluator(state=state, verdict=verdict)
        plan = self.__plan(draft=self.__draft(active=state.get_current_sub_goal()))
        return evaluator, plan, state

    async def test_command_execution_produces_record_with_real_receipt(self) -> None:
        """
        A command without postcondition finalizes at CompletionNode with its real receipt and an advancing candidate.
        """

        evaluator, plan, _ = self.__setup(
            success=self.__command(), verdict=CriterionVerdict.SATISFIED
        )
        records = await self.__records(
            evaluator=evaluator, plan=plan, receipt=self.__command_receipt()
        )
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0]["execution"]["receipt"])
        self.assertEqual(
            records[0]["post_dispatch"]["phase"]["candidate"]["kind"], AdvanceKind.ADVANCE.value
        )

    async def test_capture_execution_produces_record_with_committed_receipt(self) -> None:
        """
        A capture finalizes with its committed STORE receipt and an advancing candidate.
        """

        success = CaptureSuccess(
            target=CaptureIdentity(name="price", provenance=CaptureNameProvenance.USER),
            subject="the price",
        )
        evaluator, plan, _ = self.__setup(success=success, verdict=CriterionVerdict.SATISFIED)
        records = await self.__records(
            evaluator=evaluator, plan=plan, receipt=self.__capture_receipt()
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["execution"]["receipt"]["capture"]["name"], "price")
        self.assertEqual(
            records[0]["post_dispatch"]["phase"]["candidate"]["kind"], AdvanceKind.ADVANCE.value
        )

    async def test_command_postcondition_creates_checkpointed_proof(self) -> None:
        """
        A command with an unmet postcondition stores pending proof and retains at CompletionNode.
        """

        evaluator, plan, state = self.__setup(
            success=self.__command(postcondition="dialog closed"),
            verdict=CriterionVerdict.UNSATISFIED,
        )
        records = await self.__records(
            evaluator=evaluator, plan=plan, receipt=self.__command_receipt()
        )
        self.assertEqual(
            records[0]["post_dispatch"]["phase"]["candidate"]["kind"], AdvanceKind.RETAIN.value
        )
        proof = state.get_current_sub_goal().progress.proof
        self.assertIsNotNone(proof)
        restored = AgentState.from_checkpoint(
            state.to_checkpoint(),
            capabilities=RuntimeCapabilities(hitl=HITLCapability(enabled=False)),
        )
        self.assertIsNotNone(restored.get_current_sub_goal().progress.proof)

    async def test_stored_proof_is_consumed_on_a_later_assessed_screen(self) -> None:
        """
        With pending proof, a later pre-dispatch draft whose assessment satisfies the postcondition advances.
        """

        success = self.__command(postcondition="dialog closed")
        state = self.__state(success=success)
        evaluator = self.__evaluator(state=state, verdict=CriterionVerdict.UNSATISFIED)
        await self.__records(
            evaluator=evaluator,
            plan=self.__plan(draft=self.__draft(active=state.get_current_sub_goal())),
            receipt=self.__command_receipt(),
        )
        active = state.get_current_sub_goal()
        self.assertIsNotNone(active.progress.proof)

        analysis = AnalysisResult(
            reasoning="r",
            screen_description="s",
            visual_assessment=VisualAssessment(
                verdict=VisualVerdict.SATISFIED, confidence=0.9, evidence="closed"
            ),
            planner=PlannerMetrics(latency=1.0, calls=1),
        )
        later = ShadowRunner().draft(
            workflow_id="wf",
            active=active,
            analysis=analysis,
            metrics=PlannerMetrics(latency=1.0, calls=1),
            screen="later-hash",
            foreground="app",
            authority=TargetAuthority.unbound(),
            live_pre=Advancement(kind=AdvanceKind.RETAIN),
            cursor_before=GoalCursor(index=0, total=2),
        )
        self.assertIs(later.pre_dispatch.candidate.kind, AdvanceKind.ADVANCE)

    async def test_failed_execution_creates_no_proof(self) -> None:
        """
        A non-matching or failed receipt never stashes pending proof.
        """

        evaluator, plan, state = self.__setup(
            success=self.__command(postcondition="dialog closed"),
            verdict=CriterionVerdict.UNSATISFIED,
        )
        await self.__records(
            evaluator=evaluator,
            plan=plan,
            receipt=self.__command_receipt(target="Logout"),
        )
        self.assertIsNone(state.get_current_sub_goal().progress.proof)

    async def test_exactly_one_finalized_record_per_decision_turn(self) -> None:
        """
        One completion turn emits exactly one finalized record.
        """

        evaluator, plan, _ = self.__setup(
            success=self.__command(), verdict=CriterionVerdict.SATISFIED
        )
        records = await self.__records(
            evaluator=evaluator, plan=plan, receipt=self.__command_receipt()
        )
        self.assertEqual(len(records), 1)

    async def test_failed_execution_still_finalizes_a_non_comparable_record(self) -> None:
        """
        A failed dispatch still finalizes one record whose short-circuited live decision is not comparable.
        """

        evaluator, plan, _ = self.__setup(
            success=self.__command(), verdict=CriterionVerdict.SATISFIED
        )
        records = await self.__records(
            evaluator=evaluator,
            plan=plan,
            receipt=self.__command_receipt(executed=False, success=False),
            observation=self.__observation(),
        )
        self.assertEqual(len(records), 1)
        self.assertIsNotNone(records[0]["execution"]["receipt"])
        self.assertEqual(
            records[0]["post_dispatch"]["phase"]["kind"], PhaseComparison.INCOMPARABLE.value
        )
        self.assertEqual(records[0]["post_dispatch"]["phase"]["reason"], "EXECUTION_FAILED")

    async def test_post_screen_and_foreground_are_persisted(self) -> None:
        """
        The finalized record persists the post-dispatch screen and foreground, distinct from the pre-dispatch screen.
        """

        evaluator, plan, _ = self.__setup(
            success=self.__command(), verdict=CriterionVerdict.SATISFIED
        )
        records = await self.__records(
            evaluator=evaluator,
            plan=plan,
            receipt=self.__command_receipt(),
            observation=self.__observation(),
        )
        self.assertEqual(records[0]["post_dispatch"]["screen"], "post-dispatch-hash")
        self.assertEqual(records[0]["post_dispatch"]["foreground"], "app")
        self.assertEqual(records[0]["observation"]["screen"], "pre-dispatch-hash")

    async def test_missing_post_observation_leaves_post_screen_absent(self) -> None:
        """
        With no post observation, the post-dispatch screen is absent, never the pre-dispatch screen.
        """

        evaluator, plan, _ = self.__setup(
            success=self.__command(), verdict=CriterionVerdict.SATISFIED
        )
        records = await self.__records(
            evaluator=evaluator, plan=plan, receipt=self.__command_receipt(), observation=None
        )
        self.assertIsNone(records[0]["post_dispatch"]["screen"])
        self.assertEqual(records[0]["observation"]["screen"], "pre-dispatch-hash")

    async def test_receipt_only_goal_post_is_comparable(self) -> None:
        """
        A command without a postcondition proves from its receipt, so the post-dispatch phase is comparable.
        """

        evaluator, plan, _ = self.__setup(
            success=self.__command(), verdict=CriterionVerdict.SATISFIED
        )
        records = await self.__records(
            evaluator=evaluator,
            plan=plan,
            receipt=self.__command_receipt(),
            observation=self.__observation(),
        )
        self.assertEqual(
            records[0]["post_dispatch"]["phase"]["kind"], PhaseComparison.COMPARABLE.value
        )

    async def test_observed_goal_post_is_comparable_with_real_assessment(self) -> None:
        """
        An observed goal produces a real post-action vision verdict, so its post-dispatch phase is comparable.
        """

        state = self.__state(success=SuccessFixtures.observed())
        evaluator = self.__evaluator(
            state=state,
            verdict=CriterionVerdict.SATISFIED,
            outcome=_StubOutcome(verdict=VisualVerdict.SATISFIED),
        )
        plan = self.__plan(draft=self.__draft(active=state.get_current_sub_goal()))
        records = await self.__records(
            evaluator=evaluator,
            plan=plan,
            receipt=self.__observed_receipt(),
            observation=self.__observation(),
        )
        self.assertEqual(len(records), 1)
        phase = records[0]["post_dispatch"]["phase"]
        self.assertEqual(phase["kind"], PhaseComparison.COMPARABLE.value)
        self.assertEqual(phase["candidate"]["kind"], AdvanceKind.ADVANCE.value)

    async def test_observed_goal_stays_deferred_without_post_action_images(self) -> None:
        """
        With no post-action screen bytes to judge, an observed goal keeps the deferred, non-comparable phase.
        """

        state = self.__state(success=SuccessFixtures.observed())
        evaluator = self.__evaluator(
            state=state,
            verdict=CriterionVerdict.UNSATISFIED,
            outcome=_StubOutcome(verdict=VisualVerdict.SATISFIED),
        )
        plan = self.__plan(draft=self.__draft(active=state.get_current_sub_goal()))
        records = await self.__records(
            evaluator=evaluator,
            plan=plan,
            receipt=self.__observed_receipt(artifacts=False),
            observation=self.__observation(),
        )
        self.assertEqual(records[0]["post_dispatch"]["phase"]["reason"], "VISUAL_EVIDENCE_DEFERRED")

    async def test_visual_goal_post_is_not_comparable(self) -> None:
        """
        A command with a postcondition needs a later assessment, so its post-dispatch phase is not comparable.
        """

        evaluator, plan, _ = self.__setup(
            success=self.__command(postcondition="dialog closed"),
            verdict=CriterionVerdict.UNSATISFIED,
        )
        records = await self.__records(
            evaluator=evaluator,
            plan=plan,
            receipt=self.__command_receipt(),
            observation=self.__observation(),
        )
        self.assertEqual(
            records[0]["post_dispatch"]["phase"]["kind"], PhaseComparison.INCOMPARABLE.value
        )
        self.assertEqual(records[0]["post_dispatch"]["phase"]["reason"], "VISUAL_EVIDENCE_DEFERRED")
