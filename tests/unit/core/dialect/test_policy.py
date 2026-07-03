from __future__ import annotations

import unittest
from typing import FrozenSet, Optional, Tuple

from fathom.constants.execution import LAUNCHER_PACKAGES
from fathom.constants.flow import CheckKind, IssueCode, LaunchProvenance, ScrollDirection
from fathom.core.dialect.policy import Policy
from fathom.schemas.flow import (
    BranchNode,
    Check,
    CheckNode,
    Evidence,
    EvidenceStep,
    Flow,
    FlowNode,
    Guard,
    LaunchNode,
    ScrollNode,
    ScrollUntilNode,
    Selector,
    StepCapture,
    StepGuard,
    StepLaunch,
    StepTarget,
    StepWait,
    StoreNode,
    TapNode,
    TypeNode,
    WaitNode,
)
from fathom.schemas.steps import StepGoal


class PolicyTest(unittest.TestCase):
    """
    Cover the flow-vs-evidence fidelity gate.
    """

    def setUp(self) -> None:
        """
        Build a shared policy and recorded evidence with launcher, overlay, and recovery steps.
        """

        self.__policy = Policy()
        self.__evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1, event="action", action="tap", target=StepTarget(export="Login")
                ),
                EvidenceStep(
                    index=2,
                    action="tap",
                    event="action",
                    guard=StepGuard(condition="Overlay is visible", conditional=True, overlay=True),
                ),
                EvidenceStep(
                    index=3,
                    action="back",
                    event="action",
                    guard=StepGuard(condition="recovery"),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )

    def __codes(
        self,
        *,
        nodes: Tuple[FlowNode, ...],
        partial: bool = False,
        evidence: Optional[Evidence] = None,
    ) -> FrozenSet[IssueCode]:
        """
        Evaluate a flow and return the set of raised issue codes.
        """

        flow = Flow(intent="open and verify", package="com.example", nodes=nodes, partial=partial)
        report = self.__policy.evaluate(flow=flow, evidence=evidence or self.__evidence)

        return frozenset(issue.code for issue in report.issues)

    def __partial_evidence(self) -> Evidence:
        """
        Build evidence for a run that recorded no validation step.
        """

        return Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            partial=True,
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(index=1, event="action", action="tap", target=StepTarget(export="B")),
            ),
        )

    def __launch_only_evidence(
        self,
        *,
        package: str,
        provenance: LaunchProvenance,
        source_steps: Tuple[int, ...] = (0,),
    ) -> Evidence:
        """
        Build evidence carrying a single launch marker plus a tap and a validation step.
        """

        return Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package=package, provenance=provenance, source_steps=source_steps
                    ),
                ),
                EvidenceStep(
                    index=1, event="action", action="tap", target=StepTarget(export="Login")
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )

    def __scroll_evidence(
        self, *, scroll_target: Optional[str], rationale: Optional[str] = None
    ) -> Evidence:
        """
        Build evidence whose step 1 is a scroll, optionally toward a recorded target.
        """

        return Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="scroll",
                    rationale=rationale,
                    target=StepTarget(scroll=scroll_target),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )

    def __launch(self, *, package: str = "com.example") -> LaunchNode:
        """
        Build a synthetic warm-start launch node matching the fixture's warm-start marker.
        """

        return LaunchNode(package=package, source_steps=())

    def __tap(self, *, step: int) -> TapNode:
        """
        Build a grounded tap node bound to the given evidence step.
        """

        return TapNode(selector=Selector(text="Login"), source_steps=(step,))

    def __terminal(self) -> CheckNode:
        """
        Build a grounded terminal validation node.
        """

        return CheckNode(checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(4,))

    def test_grounded_flow_passes(self) -> None:
        """
        A launch-first, evidence-backed, terminally-validated flow raises no issues.
        """

        nodes: Tuple[FlowNode, ...] = (self.__launch(), self.__tap(step=1), self.__terminal())

        self.assertEqual(self.__codes(nodes=nodes), frozenset())

    def test_recovery_node_is_rejected(self) -> None:
        """
        A node derived from a recovery-marked step is rejected.
        """

        nodes: Tuple[FlowNode, ...] = (self.__tap(step=3), self.__terminal())

        self.assertIn(IssueCode.RECOVERY_NODE, self.__codes(nodes=nodes))

    def test_ungrounded_condition_is_rejected(self) -> None:
        """
        A branch guard whose condition is absent from the cited evidence is rejected.
        """

        branch = BranchNode(
            guard=Guard(condition="An overlay is visible to sign in", source_step=2),
            body=(self.__tap(step=2),),
            source_steps=(2,),
        )
        nodes: Tuple[FlowNode, ...] = (branch, self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_CONDITION, self.__codes(nodes=nodes))

    def test_grounded_condition_passes(self) -> None:
        """
        A branch guard matching the recorded condition raises no condition issue.
        """

        branch = BranchNode(
            guard=Guard(condition="Overlay is visible", source_step=2),
            body=(self.__tap(step=2),),
            source_steps=(2,),
        )
        nodes: Tuple[FlowNode, ...] = (branch, self.__terminal())

        self.assertNotIn(IssueCode.UNGROUNDED_CONDITION, self.__codes(nodes=nodes))

    def test_condition_grounded_in_rationale_passes(self) -> None:
        """
        A branch guard may use a better condition phrase from the cited step rationale.
        """

        evidence = self.__evidence.model_copy(
            update={
                "steps": (
                    *self.__evidence.steps[:2],
                    EvidenceStep(
                        index=2,
                        action="tap",
                        event="action",
                        rationale="Dismiss the Google account picker popup before login.",
                        guard=StepGuard(
                            condition="Overlay is visible", conditional=True, overlay=True
                        ),
                    ),
                    *self.__evidence.steps[3:],
                )
            }
        )
        branch = BranchNode(
            guard=Guard(condition="Google account picker popup", source_step=2),
            body=(self.__tap(step=2),),
            source_steps=(2,),
        )
        nodes: Tuple[FlowNode, ...] = (branch, self.__terminal())

        self.assertNotIn(
            IssueCode.UNGROUNDED_CONDITION, self.__codes(nodes=nodes, evidence=evidence)
        )

    def test_condition_grounded_in_target_visibility_passes(self) -> None:
        """
        A branch guard may use the cited step target as a visible condition.
        """

        evidence = self.__evidence.model_copy(
            update={
                "steps": (
                    *self.__evidence.steps[:2],
                    EvidenceStep(
                        index=2,
                        action="tap",
                        event="action",
                        target=StepTarget(export="NONE OF THE ABOVE"),
                        rationale="Dismiss the Google account picker popup before login.",
                        guard=StepGuard(
                            condition="Overlay is visible", conditional=True, overlay=True
                        ),
                    ),
                    *self.__evidence.steps[3:],
                )
            }
        )
        branch = BranchNode(
            guard=Guard(condition="NONE OF THE ABOVE is visible", source_step=2),
            body=(TapNode(selector=Selector(text="NONE OF THE ABOVE"), source_steps=(2,)),),
            source_steps=(2,),
        )
        nodes: Tuple[FlowNode, ...] = (branch, self.__terminal())

        self.assertNotIn(
            IssueCode.UNGROUNDED_CONDITION, self.__codes(nodes=nodes, evidence=evidence)
        )

    def test_raw_condition_is_rejected_when_narrative_condition_exists(self) -> None:
        """
        A generic recorded guard cannot override a more specific recorded rationale.
        """

        evidence = self.__evidence.model_copy(
            update={
                "steps": (
                    *self.__evidence.steps[:2],
                    EvidenceStep(
                        index=2,
                        action="tap",
                        event="action",
                        rationale="Dismiss the Google account picker popup before login.",
                        guard=StepGuard(
                            condition="Overlay is visible", conditional=True, overlay=True
                        ),
                    ),
                    *self.__evidence.steps[3:],
                )
            }
        )
        branch = BranchNode(
            guard=Guard(condition="Overlay is visible", source_step=2),
            body=(self.__tap(step=2),),
            source_steps=(2,),
        )
        nodes: Tuple[FlowNode, ...] = (branch, self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_CONDITION, self.__codes(nodes=nodes, evidence=evidence))

    def test_unguarded_conditional_is_rejected(self) -> None:
        """
        A top-level node citing a conditional step but not inside a branch is rejected.
        """

        nodes: Tuple[FlowNode, ...] = (self.__launch(), self.__tap(step=2), self.__terminal())

        self.assertIn(IssueCode.UNGUARDED_CONDITIONAL, self.__codes(nodes=nodes))

    def test_guarded_conditional_passes(self) -> None:
        """
        A node citing a conditional step raises no guard issue when inside an IF branch.
        """

        branch = BranchNode(
            guard=Guard(condition="Overlay is visible", source_step=2),
            body=(self.__tap(step=2),),
            source_steps=(2,),
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), branch, self.__terminal())

        self.assertNotIn(IssueCode.UNGUARDED_CONDITIONAL, self.__codes(nodes=nodes))

    def test_consecutive_branches_sharing_condition_are_rejected(self) -> None:
        """
        Two adjacent branches with the same condition must be merged into one IF block.
        """

        guard = Guard(condition="Overlay is visible", source_step=2)
        first = BranchNode(guard=guard, body=(self.__tap(step=2),), source_steps=(2,))
        second = BranchNode(guard=guard, body=(self.__tap(step=2),), source_steps=(2,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), first, second, self.__terminal())

        self.assertIn(IssueCode.REDUNDANT_BRANCH, self.__codes(nodes=nodes))

    def test_consecutive_identical_waits_are_rejected(self) -> None:
        """
        A wait that duplicates the one immediately before it is rejected.
        """

        wait = WaitNode(duration=5, source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), wait, wait, self.__terminal())

        self.assertIn(IssueCode.REDUNDANT_WAIT, self.__codes(nodes=nodes))

    def test_distinct_consecutive_waits_pass(self) -> None:
        """
        Two adjacent waits with different forms are not treated as duplicates.
        """

        first = WaitNode(duration=5, source_steps=(1,))
        second = WaitNode(subject="home", source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), first, second, self.__terminal())

        self.assertNotIn(IssueCode.REDUNDANT_WAIT, self.__codes(nodes=nodes))

    def test_dangling_provenance_is_rejected(self) -> None:
        """
        A node citing a step absent from the evidence is rejected.
        """

        nodes: Tuple[FlowNode, ...] = (self.__tap(step=999), self.__terminal())

        self.assertIn(IssueCode.DANGLING_PROVENANCE, self.__codes(nodes=nodes))

    def test_terminal_validation_must_cite_a_recorded_validation_step(self) -> None:
        """
        A complete run whose terminal check cites no recorded validation step is rejected.
        """

        check = CheckNode(
            checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(1,)
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), self.__tap(step=1), check)

        self.assertIn(IssueCode.INVENTED_VALIDATION, self.__codes(nodes=nodes))

    def test_partial_evidence_requires_partial_flow(self) -> None:
        """
        Partial evidence with the flow's partial flag unset is rejected.
        """

        nodes: Tuple[FlowNode, ...] = (self.__launch(), self.__tap(step=1))

        codes = self.__codes(nodes=nodes, partial=False, evidence=self.__partial_evidence())

        self.assertIn(IssueCode.MISSING_PARTIAL, codes)

    def test_partial_flow_needs_no_terminal_validation(self) -> None:
        """
        A partial run with the flag set and no invented terminal validation passes.
        """

        nodes: Tuple[FlowNode, ...] = (self.__launch(), self.__tap(step=1))

        codes = self.__codes(nodes=nodes, partial=True, evidence=self.__partial_evidence())

        self.assertNotIn(IssueCode.MISSING_PARTIAL, codes)
        self.assertNotIn(IssueCode.MISSING_GOAL_VALIDATION, codes)
        self.assertNotIn(IssueCode.INVENTED_VALIDATION, codes)

    def test_missing_terminal_validation_is_rejected(self) -> None:
        """
        A flow not ending in a check is rejected.
        """

        nodes: Tuple[FlowNode, ...] = (self.__tap(step=1),)

        self.assertIn(IssueCode.MISSING_GOAL_VALIDATION, self.__codes(nodes=nodes))

    def test_stray_warm_start_launch_is_rejected(self) -> None:
        """
        A second warm-start launch is rejected; warm start may appear only as the first launch.
        """

        nodes: Tuple[FlowNode, ...] = (
            self.__launch(),
            self.__tap(step=1),
            self.__launch(),
            self.__terminal(),
        )

        self.assertIn(IssueCode.STRAY_LAUNCH, self.__codes(nodes=nodes))

    def test_launch_package_mismatch_is_rejected(self) -> None:
        """
        A launch whose package differs from the normalized marker is rejected.
        """

        nodes: Tuple[FlowNode, ...] = (self.__launch(package="com.other"), self.__terminal())

        self.assertIn(IssueCode.LAUNCH_MISMATCH, self.__codes(nodes=nodes))

    def test_missing_launch_is_rejected(self) -> None:
        """
        A flow that does not begin with a launch while a marker exists is rejected.
        """

        nodes: Tuple[FlowNode, ...] = (self.__tap(step=1), self.__terminal())

        self.assertIn(IssueCode.MISSING_LAUNCH, self.__codes(nodes=nodes))

    def test_launcher_target_launch_is_rejected(self) -> None:
        """
        A launch whose target package is a launcher is rejected even when a marker is present.
        """

        launcher = sorted(LAUNCHER_PACKAGES)[0]
        evidence = self.__launch_only_evidence(
            package=launcher, provenance=LaunchProvenance.LAUNCHER_TRANSITION
        )
        launch = LaunchNode(
            package=launcher,
            provenance=LaunchProvenance.LAUNCHER_TRANSITION,
            source_steps=(0,),
        )
        nodes: Tuple[FlowNode, ...] = (launch, self.__tap(step=1), self.__terminal())

        self.assertIn(IssueCode.LAUNCH_MISMATCH, self.__codes(nodes=nodes, evidence=evidence))

    def test_ungrounded_transition_launch_is_rejected(self) -> None:
        """
        A launcher-transition launch citing no collapsed launcher steps is rejected.
        """

        evidence = self.__launch_only_evidence(
            package="com.example", provenance=LaunchProvenance.LAUNCHER_TRANSITION
        )
        launch = LaunchNode(package="com.example", provenance=LaunchProvenance.LAUNCHER_TRANSITION)
        nodes: Tuple[FlowNode, ...] = (launch, self.__tap(step=1), self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_LAUNCH, self.__codes(nodes=nodes, evidence=evidence))

    def test_multiple_marker_backed_launches_pass(self) -> None:
        """
        Two launches matching two ordered markers in package and provenance raise no launch issue.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1, event="action", action="tap", target=StepTarget(export="Login")
                ),
                EvidenceStep(
                    index=2,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.other",
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                        source_steps=(2,),
                    ),
                ),
                EvidenceStep(
                    index=3,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )
        second = LaunchNode(
            package="com.other",
            provenance=LaunchProvenance.LAUNCHER_TRANSITION,
            source_steps=(2,),
        )
        terminal = CheckNode(
            checks=(Check(kind=CheckKind.VISIBLE, subject="home"),), source_steps=(3,)
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), self.__tap(step=1), second, terminal)

        codes = self.__codes(nodes=nodes, evidence=evidence)

        self.assertNotIn(IssueCode.LAUNCH_MISMATCH, codes)
        self.assertNotIn(IssueCode.STRAY_LAUNCH, codes)
        self.assertNotIn(IssueCode.UNGROUNDED_LAUNCH, codes)

    def test_launch_is_exempt_from_provenance(self) -> None:
        """
        A launch matching its marker is not provenance-checked, even citing a step absent from steps.
        """

        evidence = self.__launch_only_evidence(
            package="com.example",
            provenance=LaunchProvenance.LAUNCHER_TRANSITION,
            source_steps=(999,),
        )
        launch = LaunchNode(
            package="com.example",
            provenance=LaunchProvenance.LAUNCHER_TRANSITION,
            source_steps=(999,),
        )
        nodes: Tuple[FlowNode, ...] = (launch, self.__tap(step=1), self.__terminal())

        self.assertNotIn(
            IssueCode.DANGLING_PROVENANCE, self.__codes(nodes=nodes, evidence=evidence)
        )

    def test_launch_source_steps_must_match_marker_exactly(self) -> None:
        """
        A launch whose source_steps differ from its marker's is rejected, not silently accepted.
        """

        evidence = self.__launch_only_evidence(
            package="com.example",
            provenance=LaunchProvenance.LAUNCHER_TRANSITION,
            source_steps=(0,),
        )
        launch = LaunchNode(
            package="com.example",
            provenance=LaunchProvenance.LAUNCHER_TRANSITION,
            source_steps=(999,),
        )
        nodes: Tuple[FlowNode, ...] = (launch, self.__tap(step=1), self.__terminal())

        self.assertIn(IssueCode.LAUNCH_MISMATCH, self.__codes(nodes=nodes, evidence=evidence))

    def test_bare_scroll_toward_a_recorded_target_passes(self) -> None:
        """
        A plain scroll whose recorded step scrolled toward a target is faithful.
        """

        evidence = self.__scroll_evidence(scroll_target="More products")
        scroll = ScrollNode(direction=ScrollDirection.DOWN, source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), scroll, self.__terminal())

        self.assertNotIn(IssueCode.UNGROUNDED_SCROLL, self.__codes(nodes=nodes, evidence=evidence))

    def test_scroll_until_a_recorded_target_passes(self) -> None:
        """
        A scroll-until node citing the recorded scroll target raises no scroll issue.
        """

        evidence = self.__scroll_evidence(scroll_target="More products")
        scroll = ScrollUntilNode(
            direction=ScrollDirection.DOWN, target="More products", source_steps=(1,)
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), scroll, self.__terminal())

        self.assertNotIn(IssueCode.UNGROUNDED_SCROLL, self.__codes(nodes=nodes, evidence=evidence))

    def test_scroll_until_grounded_in_rationale_passes(self) -> None:
        """
        A scroll-until target may come from the cited gesture rationale.
        """

        evidence = self.__scroll_evidence(
            scroll_target="product list",
            rationale="Scroll down to find a Ghar soap with rating >= 4.2.",
        )
        scroll = ScrollUntilNode(
            direction=ScrollDirection.DOWN,
            target="Ghar soap with rating >= 4.2",
            source_steps=(1,),
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), scroll, self.__terminal())

        self.assertNotIn(IssueCode.UNGROUNDED_SCROLL, self.__codes(nodes=nodes, evidence=evidence))

    def test_consecutive_scrolls_in_one_goal_are_redundant(self) -> None:
        """
        Consecutive same-direction scrolls inside one recorded goal must be folded.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="scroll",
                    goal=StepGoal(index=2, description="Find rated product", directive="validate"),
                ),
                EvidenceStep(
                    index=2,
                    event="action",
                    action="scroll",
                    goal=StepGoal(index=2, description="Find rated product", directive="validate"),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )
        first = ScrollNode(direction=ScrollDirection.DOWN, source_steps=(1,))
        second = ScrollNode(direction=ScrollDirection.DOWN, source_steps=(2,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), first, second, self.__terminal())

        self.assertIn(IssueCode.REDUNDANT_SCROLL, self.__codes(nodes=nodes, evidence=evidence))

    def test_invented_scroll_until_target_is_rejected(self) -> None:
        """
        A scroll-until target must match the recorded scroll target on the cited gesture step.
        """

        evidence = self.__scroll_evidence(scroll_target="More products")
        scroll = ScrollUntilNode(
            direction=ScrollDirection.DOWN, target="Payment section", source_steps=(1,)
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), scroll, self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_SCROLL, self.__codes(nodes=nodes, evidence=evidence))

    def test_scroll_direction_mismatch_is_rejected(self) -> None:
        """
        A scroll direction must match the recorded gesture's Drizz page-motion direction.
        """

        evidence = self.__scroll_evidence(scroll_target=None)
        scroll = ScrollNode(direction=ScrollDirection.UP, source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), scroll, self.__terminal())

        self.assertIn(
            IssueCode.SCROLL_DIRECTION_MISMATCH, self.__codes(nodes=nodes, evidence=evidence)
        )

    def test_scroll_without_a_recorded_gesture_is_rejected(self) -> None:
        """
        A scroll node cannot cite a non-gesture source step.
        """

        scroll = ScrollNode(direction=ScrollDirection.DOWN, source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), scroll, self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_SCROLL, self.__codes(nodes=nodes))

    def test_bare_scroll_without_a_recorded_target_passes(self) -> None:
        """
        A plain scroll whose recorded step has no scroll target is left alone.
        """

        evidence = self.__scroll_evidence(scroll_target=None)
        scroll = ScrollNode(direction=ScrollDirection.DOWN, source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), scroll, self.__terminal())

        self.assertNotIn(IssueCode.UNGROUNDED_SCROLL, self.__codes(nodes=nodes, evidence=evidence))

    def __store_evidence(self, *, captured: bool, success: bool = True) -> Evidence:
        """
        Build evidence whose step 1 is a store, optionally with a recorded capture of the given outcome.
        """

        capture = (
            StepCapture(
                name="total",
                subject="cart total",
                success=success,
                value="₹499" if success else None,
                reason=None if success else "no element",
            )
            if captured
            else None
        )
        return Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(index=1, event="action", action="store", capture=capture),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )

    def test_store_grounded_in_a_recorded_capture_passes(self) -> None:
        """
        A store citing a step whose evidence recorded a capture raises no store issue.
        """

        evidence = self.__store_evidence(captured=True)
        store = StoreNode(value="₹499", name="total", source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), store, self.__terminal())

        self.assertNotIn(IssueCode.UNGROUNDED_STORE, self.__codes(nodes=nodes, evidence=evidence))

    def test_store_without_a_recorded_capture_is_rejected(self) -> None:
        """
        A store citing a step that recorded no capture is rejected as ungrounded.
        """

        evidence = self.__store_evidence(captured=False)
        store = StoreNode(value="₹499", name="total", source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), store, self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_STORE, self.__codes(nodes=nodes, evidence=evidence))

    def test_tap_target_mismatch_is_rejected(self) -> None:
        """
        A tap citing the right source step is rejected when its target text is invented.
        """

        tap = TapNode(selector=Selector(text="Create account"), source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), tap, self.__terminal())

        self.assertIn(IssueCode.TAP_TARGET_MISMATCH, self.__codes(nodes=nodes))

    def test_tap_may_use_rationale_grounded_target(self) -> None:
        """
        A tap target can be authored from the cited action rationale.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    rationale="Tap the search box to gain focus.",
                    target=StepTarget(
                        name="Search by Keyword or Product ID",
                        export="Search by Keyword or Product ID search bar",
                    ),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )
        tap = TapNode(selector=Selector(text="search box"), source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), tap, self.__terminal())

        self.assertNotIn(
            IssueCode.TAP_TARGET_MISMATCH, self.__codes(nodes=nodes, evidence=evidence)
        )

    def test_type_content_mismatch_is_rejected(self) -> None:
        """
        A type node citing the right source step is rejected when field or typed text differ.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="type",
                    text="aman",
                    target=StepTarget(export="Name field"),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )
        typed = TypeNode(text="rohan", field=Selector(text="Name field"), source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), typed, self.__terminal())

        self.assertIn(IssueCode.TYPE_CONTENT_MISMATCH, self.__codes(nodes=nodes, evidence=evidence))

    def test_wait_subject_mismatch_is_rejected(self) -> None:
        """
        A wait node citing the right source step is rejected when its subject differs.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=1,
                    event="action",
                    action="wait",
                    wait=StepWait(subject="Home"),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="home"),
                ),
            ),
        )
        wait = WaitNode(subject="Profile", source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), wait, self.__terminal())

        self.assertIn(IssueCode.WAIT_SUBJECT_MISMATCH, self.__codes(nodes=nodes, evidence=evidence))

    def test_validation_subject_mismatch_is_rejected(self) -> None:
        """
        A validation node citing a recorded validation cannot assert a different subject.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    target=StepTarget(export="Home screen"),
                ),
            ),
        )
        terminal = CheckNode(
            checks=(Check(kind=CheckKind.VISIBLE, subject="Cart screen"),), source_steps=(4,)
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), terminal)

        self.assertIn(
            IssueCode.VALIDATION_SUBJECT_MISMATCH,
            self.__codes(nodes=nodes, evidence=evidence),
        )

    def test_validation_rejects_incidental_target_when_assertion_state_is_recorded(self) -> None:
        """
        A validation cannot use incidental anchor text when the assertion state was recorded.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    rationale="The login screen is visible.",
                    target=StepTarget(name="Phone Number", export="Phone Number input field"),
                ),
            ),
        )
        terminal = CheckNode(
            checks=(Check(kind=CheckKind.VISIBLE, subject="Phone Number"),), source_steps=(4,)
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), terminal)

        self.assertIn(
            IssueCode.VALIDATION_SUBJECT_MISMATCH,
            self.__codes(nodes=nodes, evidence=evidence),
        )

    def test_validation_accepts_state_grounded_in_rationale(self) -> None:
        """
        A validation may assert the state named by the cited validation rationale.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(
                    index=4,
                    event="validation",
                    action="complete",
                    rationale="The login screen is visible.",
                    target=StepTarget(name="Phone Number", export="Phone Number input field"),
                ),
            ),
        )
        terminal = CheckNode(
            checks=(Check(kind=CheckKind.VISIBLE, subject="login screen"),), source_steps=(4,)
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), terminal)

        self.assertNotIn(
            IssueCode.VALIDATION_SUBJECT_MISMATCH,
            self.__codes(nodes=nodes, evidence=evidence),
        )

    def test_validation_without_recorded_target_is_rejected(self) -> None:
        """
        A validation step with no recorded target cannot ground an authored check subject.
        """

        evidence = Evidence(
            goal="home visible",
            package="com.example",
            intent="open and verify",
            steps=(
                EvidenceStep(
                    index=0,
                    action="launch",
                    event="launch",
                    launch=StepLaunch(
                        package="com.example",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(index=4, event="validation", action="complete"),
            ),
        )
        terminal = CheckNode(
            checks=(Check(kind=CheckKind.VISIBLE, subject="Home screen"),), source_steps=(4,)
        )
        nodes: Tuple[FlowNode, ...] = (self.__launch(), terminal)

        self.assertIn(
            IssueCode.VALIDATION_SUBJECT_MISMATCH,
            self.__codes(nodes=nodes, evidence=evidence),
        )

    def test_store_with_wrong_name_is_rejected(self) -> None:
        """
        A store whose variable name does not match the recorded capture is rejected.
        """

        evidence = self.__store_evidence(captured=True)
        store = StoreNode(value="₹499", name="coupon", source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), store, self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_STORE, self.__codes(nodes=nodes, evidence=evidence))

    def test_store_with_wrong_value_is_rejected(self) -> None:
        """
        A store whose value does not match the recorded capture is rejected.
        """

        evidence = self.__store_evidence(captured=True)
        store = StoreNode(value="₹129", name="total", source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), store, self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_STORE, self.__codes(nodes=nodes, evidence=evidence))

    def test_store_grounded_in_a_failed_capture_is_rejected(self) -> None:
        """
        A store matching a capture that failed at record time is rejected; only successful captures ground.
        """

        evidence = self.__store_evidence(captured=True, success=False)
        store = StoreNode(value="₹499", name="total", source_steps=(1,))
        nodes: Tuple[FlowNode, ...] = (self.__launch(), store, self.__terminal())

        self.assertIn(IssueCode.UNGROUNDED_STORE, self.__codes(nodes=nodes, evidence=evidence))
