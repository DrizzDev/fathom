from __future__ import annotations

import unittest
from typing import Tuple

from fathom.constants.flow import (
    AssertionSource,
    CheckKind,
    EvidenceMarker,
    LaunchProvenance,
    NodeKind,
    ScrollDirection,
)
from fathom.constants.generation import SkipReason
from fathom.core.dialect.policy import Policy
from fathom.core.services.generation.projector import DeterministicFlowGenerator
from fathom.schemas.flow import (
    BranchNode,
    CheckNode,
    CompletionAssertion,
    Evidence,
    EvidenceStep,
    ScrollNode,
    ScrollUntilNode,
    StepCapture,
    StepGuard,
    StepLaunch,
    StepOutcome,
    StepTarget,
    StoreNode,
    TapNode,
    TargetAnchors,
    TargetClaim,
    TargetStructure,
)
from fathom.schemas.steps import StepGoal


class DeterministicFlowGeneratorTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins the deterministic projection: faithful node-per-step mapping that passes the fidelity policy.
    """

    def __generator(self) -> DeterministicFlowGenerator:
        """
        Build the projector under test.
        """

        return DeterministicFlowGenerator()

    @staticmethod
    def __launch(*, index: int = 0, package: str = "in.swiggy.android") -> EvidenceStep:
        """
        Build a launcher-transition launch evidence step.
        """

        return EvidenceStep(
            index=index,
            event="launch",
            action="launch",
            launch=StepLaunch(
                package=package,
                provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                source_steps=(index,),
            ),
        )

    @staticmethod
    def __tap(*, index: int, export: str) -> EvidenceStep:
        """
        Build a successful tap evidence step grounded on an export phrase.
        """

        return EvidenceStep(
            index=index,
            event="action",
            action="tap",
            target=StepTarget(export=export),
            outcome=StepOutcome(success=True),
        )

    @staticmethod
    def __validation(*, index: int, success: bool = True) -> EvidenceStep:
        """
        Build a goal validation evidence step.
        """

        return EvidenceStep(
            index=index,
            event="validation",
            action="complete",
            target=StepTarget(export="Cart screen"),
            outcome=StepOutcome(success=success),
        )

    @staticmethod
    def __evidence(*, steps: Tuple[EvidenceStep, ...], partial: bool = False) -> Evidence:
        """
        Wrap evidence steps for the Swiggy-style run.
        """

        return Evidence(
            intent="order a burger on swiggy",
            goal="McDonald's burger added to cart",
            package="in.swiggy.android",
            steps=steps,
            partial=partial,
        )

    async def test_swiggy_run_projects_to_policy_clean_flow(self) -> None:
        """
        The Swiggy run (launch + taps + validation) projects to a launch-led, validation-terminated, policy-clean flow.
        """

        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                self.__tap(index=1, export="address field"),
                self.__tap(index=2, export="HSR Layout"),
                self.__tap(index=3, export="Search box"),
                self.__tap(index=4, export="McDonald's"),
                self.__tap(index=5, export="ADD button"),
                self.__validation(index=6),
            ),
        )

        flow = await self.__generator().generate(evidence=evidence)
        report = Policy().evaluate(flow=flow, evidence=evidence)

        self.assertEqual(report.issues, ())
        self.assertIs(flow.nodes[0].kind, NodeKind.LAUNCH)
        self.assertIs(flow.nodes[-1].kind, NodeKind.CHECK)
        self.assertFalse(flow.partial)

    async def test_successful_capture_projects_to_store_node(self) -> None:
        """
        A successful STORE step projects to a Store node carrying the captured value and name.
        """

        store_step = EvidenceStep(
            index=1,
            event="action",
            action="store",
            outcome=StepOutcome(success=True),
            capture=StepCapture(
                name="total",
                subject="cart total",
                success=True,
                value="499",
            ),
        )
        evidence = self.__evidence(
            steps=(self.__launch(index=0), store_step, self.__validation(index=2)),
        )

        flow = await self.__generator().generate(evidence=evidence)
        stores = [node for node in flow.nodes if isinstance(node, StoreNode)]

        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0].value, "499")
        self.assertEqual(stores[0].name, "total")

    async def test_failed_and_recovery_steps_are_skipped(self) -> None:
        """
        A failed step and a recovery-marked step never reach the flow.
        """

        failed = EvidenceStep(
            index=1,
            event="action",
            action="tap",
            target=StepTarget(export="broken"),
            outcome=StepOutcome(success=False),
        )
        recovery = EvidenceStep(
            index=2,
            event="action",
            action="tap",
            target=StepTarget(export="recovered"),
            outcome=StepOutcome(success=True),
            guard=StepGuard(condition=EvidenceMarker.RECOVERY),
        )
        evidence = self.__evidence(
            steps=(self.__launch(index=0), failed, recovery, self.__validation(index=3)),
        )

        flow = await self.__generator().generate(evidence=evidence)

        self.assertEqual([node.kind for node in flow.nodes], [NodeKind.LAUNCH, NodeKind.CHECK])

    async def test_projection_report_records_skipped_steps_with_reasons(self) -> None:
        """
        Failed, recovery, and unsupported steps are dropped from the flow and recorded with reasons.
        """

        failed = EvidenceStep(
            index=1,
            event="action",
            action="tap",
            target=StepTarget(export="x"),
            outcome=StepOutcome(success=False),
        )
        recovery = EvidenceStep(
            index=2,
            event="action",
            action="tap",
            target=StepTarget(export="y"),
            outcome=StepOutcome(success=True),
            guard=StepGuard(condition=EvidenceMarker.RECOVERY),
        )
        long_press = EvidenceStep(
            index=3,
            event="action",
            action="long_press",
            target=StepTarget(export="z"),
            outcome=StepOutcome(success=True),
        )
        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                failed,
                recovery,
                long_press,
                self.__validation(index=4),
            ),
        )

        report = self.__generator().project(evidence=evidence)
        reasons = {skip.index: skip.reason for skip in report.skipped}

        self.assertEqual(
            reasons,
            {1: SkipReason.FAILED, 2: SkipReason.RECOVERY, 3: SkipReason.UNSUPPORTED},
        )
        self.assertEqual(
            [node.kind for node in report.flow.nodes], [NodeKind.LAUNCH, NodeKind.CHECK]
        )

    async def test_no_validation_yields_partial_flow(self) -> None:
        """
        A run with no successful validation projects to a partial flow with no terminal check.
        """

        evidence = self.__evidence(
            steps=(self.__launch(index=0), self.__tap(index=1, export="address field")),
            partial=True,
        )

        flow = await self.__generator().generate(evidence=evidence)
        report = Policy().evaluate(flow=flow, evidence=evidence)

        self.assertTrue(flow.partial)
        self.assertNotIn(NodeKind.CHECK, [node.kind for node in flow.nodes])
        self.assertEqual(report.issues, ())

    async def test_consecutive_scrolls_in_one_goal_project_to_one_scroll(self) -> None:
        """
        Repeated scroll attempts inside one execution episode become one replay scroll.
        """

        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="swipe_up",
                    outcome=StepOutcome(success=True),
                    goal=StepGoal(index=2, description="Find rated product", directive="validate"),
                    target=StepTarget(scroll="Ratings & Reviews"),
                ),
                EvidenceStep(
                    index=2,
                    event="action",
                    action="swipe_up",
                    outcome=StepOutcome(success=True),
                    goal=StepGoal(index=2, description="Find rated product", directive="validate"),
                    target=StepTarget(scroll="product list"),
                ),
                self.__validation(index=3),
            ),
        )

        flow = await self.__generator().generate(evidence=evidence)
        scrolls = [node for node in flow.nodes if isinstance(node, ScrollNode)]

        self.assertEqual(len(scrolls), 1)
        self.assertEqual(scrolls[0].direction, ScrollDirection.DOWN)
        self.assertEqual(scrolls[0].source_steps, (1, 2))
        self.assertEqual(Policy().evaluate(flow=flow, evidence=evidence).issues, ())

    async def test_scroll_until_target_preserves_recorded_target(self) -> None:
        """
        A recorded scroll target is preserved without deterministic wording glue.
        """

        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="swipe_up",
                    target=StepTarget(scroll="Ratings & Reviews section"),
                ),
                self.__validation(index=2),
            ),
        )

        flow = await self.__generator().generate(evidence=evidence)
        scroll = next(node for node in flow.nodes if isinstance(node, ScrollUntilNode))

        self.assertEqual(scroll.target, "Ratings & Reviews section")
        self.assertEqual(Policy().evaluate(flow=flow, evidence=evidence).issues, ())

    async def test_tap_uses_verified_anchor_before_unverified_claim(self) -> None:
        """
        An unverified planner claim is not projected as the baseline tap target.
        """

        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    target=StepTarget(
                        claim=TargetClaim(text="Magic Soap 3 Pack", verified=False),
                        anchors=TargetAnchors(accessibility=("product card",)),
                        structure=TargetStructure(role="product card"),
                    ),
                ),
                self.__validation(index=2),
            ),
        )

        flow = await self.__generator().generate(evidence=evidence)
        tap = next(node for node in flow.nodes if isinstance(node, TapNode))

        self.assertEqual(tap.selector.text, "product card")
        self.assertEqual(Policy().evaluate(flow=flow, evidence=evidence).issues, ())

    async def test_unverified_claim_without_anchor_is_skipped_and_marks_partial(self) -> None:
        """
        A baseline never uses an unverified planner claim as a replay target.
        """

        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    target=StepTarget(
                        claim=TargetClaim(text="Magic Soap 3 Pack", verified=False),
                        name="Magic Soap 3 Pack",
                    ),
                ),
                self.__validation(index=2),
            ),
        )

        report = self.__generator().project(evidence=evidence)

        self.assertTrue(report.flow.partial)
        self.assertEqual(report.skipped[0].index, 1)
        self.assertIs(report.skipped[0].reason, SkipReason.MISSING_TARGET)
        self.assertFalse(
            any(
                isinstance(node, TapNode) and node.selector.text == "Magic Soap 3 Pack"
                for node in report.flow.nodes
            )
        )

    async def test_dynamic_address_claim_without_anchor_is_not_rendered(self) -> None:
        """
        Dynamic address content descriptions must not leak into deterministic tap targets.
        """

        address = (
            "Selected address is Manhattan, A108 Adams St, New York, NY 10007, "
            "USA Delivering in null minutes"
        )
        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    target=StepTarget(
                        claim=TargetClaim(text=address, verified=False),
                        name=address,
                    ),
                    outcome=StepOutcome(success=True),
                ),
                self.__validation(index=2),
            ),
        )

        report = self.__generator().project(evidence=evidence)

        self.assertTrue(report.flow.partial)
        self.assertEqual(report.skipped[0].index, 1)
        self.assertIs(report.skipped[0].reason, SkipReason.MISSING_TARGET)
        self.assertFalse(any(isinstance(node, TapNode) for node in report.flow.nodes))

    async def test_validation_without_target_yields_partial_flow(self) -> None:
        """
        A validation record without a concrete target is skipped instead of using prose fallback.
        """

        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                self.__tap(index=1, export="address field"),
                EvidenceStep(
                    index=2,
                    event="validation",
                    action="complete",
                    observation="The goal is complete.",
                    outcome=StepOutcome(success=True),
                ),
            ),
        )

        report = self.__generator().project(evidence=evidence)
        policy_evidence = evidence.model_copy(update={"partial": report.flow.partial})
        policy_report = Policy().evaluate(flow=report.flow, evidence=policy_evidence)

        self.assertTrue(report.flow.partial)
        self.assertEqual(
            {skip.index: skip.reason for skip in report.skipped}, {2: SkipReason.MISSING_TARGET}
        )
        self.assertNotIn(NodeKind.CHECK, [node.kind for node in report.flow.nodes])
        self.assertEqual(policy_report.issues, ())

    async def test_completion_assertions_project_to_terminal_validation(self) -> None:
        """
        Verifier assertions produce the baseline terminal Validate required for completed scripts.
        """

        evidence = self.__evidence(
            steps=(self.__launch(index=0), self.__tap(index=1, export="Buy Now button")),
        ).model_copy(
            update={
                "assertions": (
                    CompletionAssertion(
                        id="terminal.login",
                        kind=CheckKind.VISIBLE,
                        subject="Phone number input field",
                        source=AssertionSource.VERIFICATION,
                        step_index=1,
                    ),
                )
            }
        )

        flow = await self.__generator().generate(evidence=evidence)
        terminal = flow.nodes[-1]

        self.assertIsInstance(terminal, CheckNode)
        assert isinstance(terminal, CheckNode)
        self.assertFalse(flow.partial)
        self.assertEqual(terminal.source_steps, (1,))
        self.assertEqual(terminal.assertion_ids, ("terminal.login",))
        self.assertEqual(terminal.checks[0].subject, "Phone number input field")
        self.assertEqual(Policy().evaluate(flow=flow, evidence=evidence).issues, ())

    async def test_completion_assertion_with_missing_step_uses_last_kept_step(self) -> None:
        """
        Verifier assertions citing a non-kept step still produce valid provenance.
        """

        evidence = self.__evidence(
            steps=(self.__launch(index=0), self.__tap(index=1, export="Buy Now button")),
        ).model_copy(
            update={
                "assertions": (
                    CompletionAssertion(
                        id="terminal.login",
                        kind=CheckKind.VISIBLE,
                        source=AssertionSource.VERIFICATION,
                        subject="Phone number input field",
                        step_index=8,
                    ),
                )
            }
        )

        flow = await self.__generator().generate(evidence=evidence)
        terminal = flow.nodes[-1]

        self.assertIsInstance(terminal, CheckNode)
        assert isinstance(terminal, CheckNode)
        self.assertEqual(terminal.source_steps, (1,))
        self.assertEqual(Policy().evaluate(flow=flow, evidence=evidence).issues, ())

    async def test_conditional_target_preserves_recorded_guard_condition(self) -> None:
        """
        A guarded target action uses the recorded condition without inventing predicate text.
        """

        evidence = self.__evidence(
            steps=(
                self.__launch(index=0),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    target=StepTarget(export="NONE OF THE ABOVE"),
                    outcome=StepOutcome(success=True),
                    guard=StepGuard(condition="Overlay is visible", conditional=True),
                    rationale="Dismiss the Google account picker popup.",
                ),
                self.__validation(index=2),
            ),
        )

        flow = await self.__generator().generate(evidence=evidence)
        branches = [node for node in flow.nodes if isinstance(node, BranchNode)]

        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0].guard.condition, "Overlay is visible")
        self.assertEqual(Policy().evaluate(flow=flow, evidence=evidence).issues, ())
