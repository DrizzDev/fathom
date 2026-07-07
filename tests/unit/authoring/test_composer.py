from __future__ import annotations

import unittest
from typing import Tuple

from fathom.authoring.application.composer import StepDraftComposer
from fathom.constants.authoring import AuthoringArtifactKind, AuthoringKind, AuthoringStatus
from fathom.constants.dialect import DialectName
from fathom.constants.flow import AssertionSource, CheckKind, LaunchProvenance
from fathom.constants.generation import ScriptCommandRole, ScriptSource
from fathom.schemas.authoring import AuthoringArtifact, AuthoringBaseline, AuthoringBaselineCommand
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.flow import (
    CompletionAssertion,
    Evidence,
    EvidenceStep,
    StepGuard,
    StepLaunch,
    StepOutcome,
)
from fathom.schemas.generation import CompletionValidation


class StepDraftComposerTest(unittest.TestCase):
    """
    Cover composition of reviewed step drafts into a final fallback script.
    """

    def test_compose_uses_distilled_evidence_order_and_latest_step_draft(self) -> None:
        """
        Composed fallback follows evidence order and ignores drafts for discarded steps.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="search",
            goal="search",
            package="com.example",
            partial=True,
            discarded=(3,),
            steps=(
                EvidenceStep(index=1, event="action", action="tap"),
                EvidenceStep(index=2, event="action", action="type"),
                EvidenceStep(index=4, event="action", action="store"),
            ),
        )
        drafts: Tuple[AuthoringDraft, ...] = (
            self.__draft(step=2, text="old type"),
            self.__draft(step=1, text="OPEN_APP: com.example"),
            self.__draft(step=3, text="discarded recovery"),
            self.__draft(step=2, text='Type "soap" into search field'),
            self.__draft(step=4, text="Store 89 as product.amount"),
        )

        result = composer.compose(drafts=drafts, evidence=evidence)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, ScriptSource.STEP_DRAFTS)
        self.assertEqual(
            result.text,
            "\n".join(
                (
                    "OPEN_APP: com.example",
                    'Type "soap" into search field',
                    "Store 89 as product.amount",
                )
            ),
        )
        self.assertTrue(result.review.partial)
        self.assertEqual(result.review.discarded, (3,))
        self.assertEqual(
            [lineage.source_steps for lineage in result.review.lineage],
            [(1,), (2,), (4,)],
        )
        self.assertEqual(
            [command.source_steps for command in result.review.commands],
            [(1,), (2,), (4,)],
        )
        self.assertEqual(
            [command.verified_by for command in result.review.commands],
            [(), (), ()],
        )
        self.assertEqual(
            [lineage.screen_authored for lineage in result.review.lineage],
            [True, True, True],
        )

    def test_compose_fills_missing_step_draft_from_baseline(self) -> None:
        """
        Missing step drafts use the corresponding deterministic baseline line.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="search",
            goal="search",
            package="com.example",
            partial=True,
            steps=(
                EvidenceStep(index=1, event="action", action="tap"),
                EvidenceStep(index=2, event="action", action="type"),
                EvidenceStep(index=3, event="action", action="tap"),
            ),
        )
        baseline = AuthoringBaseline(
            content="\n".join(
                (
                    "OPEN_APP: com.example",
                    'Type "soap" into Search',
                    "Tap on product",
                )
            ),
            partial=True,
            commands=(
                AuthoringBaselineCommand(text="OPEN_APP: com.example", source_steps=(1,)),
                AuthoringBaselineCommand(text='Type "soap" into Search', source_steps=(2,)),
                AuthoringBaselineCommand(text="Tap on product", source_steps=(3,)),
            ),
        )
        drafts = (
            self.__draft(step=1, text="OPEN_APP: com.example"),
            self.__draft(step=3, text="Tap on first product card"),
        )

        result = composer.compose(drafts=drafts, evidence=evidence, baseline=baseline)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.text,
            "\n".join(
                (
                    "OPEN_APP: com.example",
                    'Type "soap" into Search',
                    "Tap on first product card",
                )
            ),
        )
        self.assertEqual(
            [command.source_steps for command in result.review.commands],
            [(1,), (2,), (3,)],
        )
        self.assertEqual(
            [command.verified_by for command in result.review.commands],
            [(), ("execution",), ()],
        )

    def test_grounded_launch_baseline_is_not_duplicated_by_step_draft(self) -> None:
        """
        STEP_DRAFTS keeps the deterministic launch scaffold and skips launch drafts.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="login",
            goal="login",
            package="com.healthtap.userhtexpress",
            partial=False,
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    launch=StepLaunch(
                        package="com.healthtap.userhtexpress",
                        provenance=LaunchProvenance.LAUNCHER_TRANSITION,
                        source_steps=(0,),
                    ),
                ),
                EvidenceStep(index=1, event="action", action="type"),
            ),
        )
        baseline = AuthoringBaseline(
            content="\n".join(
                (
                    "OPEN_APP: com.healthtap.userhtexpress",
                    'Type "user@example.com" into Email field',
                )
            ),
            partial=False,
            commands=(
                AuthoringBaselineCommand(
                    text="OPEN_APP: com.healthtap.userhtexpress",
                    role=ScriptCommandRole.LAUNCH,
                    source_steps=(0,),
                ),
                AuthoringBaselineCommand(
                    text='Type "user@example.com" into Email field',
                    source_steps=(1,),
                ),
            ),
        )
        drafts = (
            self.__draft(step=0, text="OPEN_APP: com.healthtap.userhtexpress"),
            self.__draft(step=1, text='Type "user@example.com" into Email field'),
        )

        result = composer.compose(drafts=drafts, evidence=evidence, baseline=baseline)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.text.splitlines().count("OPEN_APP: com.healthtap.userhtexpress"), 1)

    def test_warm_start_launch_baseline_is_kept_without_source_steps(self) -> None:
        """
        A synthetic launch without source steps still remains the launch scaffold.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="login",
            goal="login",
            package="com.healthtap.userhtexpress",
            partial=False,
            steps=(
                EvidenceStep(
                    index=0,
                    event="launch",
                    action="launch",
                    launch=StepLaunch(
                        package="com.healthtap.userhtexpress",
                        provenance=LaunchProvenance.SYNTHETIC_WARM_START,
                    ),
                ),
                EvidenceStep(index=1, event="action", action="type"),
            ),
        )
        baseline = AuthoringBaseline(
            content="\n".join(
                (
                    "OPEN_APP: com.healthtap.userhtexpress",
                    'Type "user@example.com" into Email field',
                )
            ),
            partial=False,
            commands=(
                AuthoringBaselineCommand(
                    text="OPEN_APP: com.healthtap.userhtexpress",
                    role=ScriptCommandRole.LAUNCH,
                ),
                AuthoringBaselineCommand(
                    text='Type "user@example.com" into Email field',
                    source_steps=(1,),
                ),
            ),
        )
        drafts = (
            self.__draft(step=0, text="OPEN_APP: com.healthtap.userhtexpress"),
            self.__draft(step=1, text='Type "user@example.com" into Email field'),
        )

        result = composer.compose(drafts=drafts, evidence=evidence, baseline=baseline)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.text,
            "\n".join(
                (
                    "OPEN_APP: com.healthtap.userhtexpress",
                    'Type "user@example.com" into Email field',
                )
            ),
        )

    def test_extra_draft_lines_do_not_inherit_step_provenance(self) -> None:
        """
        Multi-line step drafts cannot launder invented follow-up commands as executed.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="search",
            goal="search",
            package="com.example",
            partial=True,
            steps=(EvidenceStep(index=1, event="action", action="type"),),
        )
        drafts = (
            self.__draft(
                step=1,
                text="\n".join(
                    (
                        'Type "soap" into search field',
                        'Wait until "suggestions list is visible"',
                    )
                ),
            ),
        )

        result = composer.compose(drafts=drafts, evidence=evidence)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            [command.source_steps for command in result.review.commands],
            [(1,), ()],
        )
        self.assertEqual(
            [lineage.screen_authored for lineage in result.review.lineage],
            [True, True],
        )

    def test_compose_fills_missing_draft_by_source_step_not_position(self) -> None:
        """
        Missing drafts use baseline command source steps after baseline line positions drift.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="search",
            goal="search",
            package="com.example",
            partial=False,
            steps=(
                EvidenceStep(index=1, event="action", action="tap"),
                EvidenceStep(index=2, event="action", action="swipe_up"),
                EvidenceStep(index=3, event="action", action="swipe_up"),
                EvidenceStep(index=4, event="action", action="store"),
            ),
        )
        baseline = AuthoringBaseline(
            content="\n".join(
                (
                    "OPEN_APP: com.example",
                    'Scroll down until "product is visible"',
                    "Store 89 as product.amount",
                )
            ),
            partial=False,
            commands=(
                AuthoringBaselineCommand(text="OPEN_APP: com.example", source_steps=(1,)),
                AuthoringBaselineCommand(
                    text='Scroll down until "product is visible"',
                    source_steps=(2, 3),
                ),
                AuthoringBaselineCommand(text="Store 89 as product.amount", source_steps=(4,)),
            ),
        )
        drafts = (
            self.__draft(step=1, text="OPEN_APP: com.example"),
            self.__draft(step=2, text="Scroll down"),
        )

        result = composer.compose(drafts=drafts, evidence=evidence, baseline=baseline)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.text,
            "\n".join(
                (
                    "OPEN_APP: com.example",
                    'Scroll down until "product is visible"',
                    "Store 89 as product.amount",
                )
            ),
        )

    def test_guarded_baseline_command_is_not_replaced_by_bare_step_draft(self) -> None:
        """
        Conditional baseline structure is preserved over an unguarded step draft.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="login",
            goal="login",
            package="com.example",
            partial=True,
            steps=(
                EvidenceStep(index=1, event="action", action="tap"),
                EvidenceStep(index=2, event="action", action="tap"),
            ),
        )
        guarded = "\n".join(
            (
                "IF Not now popup is visible",
                "{",
                "    Tap on Not now button",
                "}",
            )
        )
        baseline = AuthoringBaseline(
            content=guarded,
            partial=True,
            commands=(
                AuthoringBaselineCommand(
                    text=guarded,
                    role=ScriptCommandRole.BRANCH,
                    source_steps=(1, 2),
                ),
            ),
        )
        drafts = (self.__draft(step=2, text="Tap on Not now button"),)

        result = composer.compose(drafts=drafts, evidence=evidence, baseline=baseline)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.text, guarded)
        self.assertEqual(result.review.commands[0].role, ScriptCommandRole.BRANCH)
        self.assertEqual(result.review.commands[0].source_steps, (1, 2))

    def test_conditional_step_draft_without_baseline_is_omitted_honestly(self) -> None:
        """
        A conditional step draft is not published as an unguarded command without baseline structure.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="login",
            goal="login",
            package="com.example",
            partial=False,
            steps=(
                EvidenceStep(
                    index=2,
                    event="action",
                    action="tap",
                    guard=StepGuard(
                        conditional=True,
                        condition="Not now popup is visible",
                    ),
                ),
            ),
        )
        drafts = (self.__draft(step=2, text="Tap on Not now button"),)

        result = composer.compose(drafts=drafts, evidence=evidence)

        self.assertIsNone(result)

    def test_compose_appends_rendered_terminal_validation(self) -> None:
        """
        Completed composed drafts append the caller-rendered terminal assertion lines.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="search",
            goal="search",
            package="com.example",
            partial=False,
            assertions=(
                CompletionAssertion(
                    id="terminal.login",
                    kind=CheckKind.VISIBLE,
                    source=AssertionSource.VERIFICATION,
                    subject="login screen",
                ),
            ),
            steps=(EvidenceStep(index=1, event="action", action="tap"),),
        )
        drafts = (self.__draft(step=1, text="OPEN_APP: com.example"),)

        result = composer.compose(
            drafts=drafts,
            evidence=evidence,
            completion=CompletionValidation(
                lines=("Validate login screen is visible",),
                required=True,
                source_steps=(1,),
            ),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.review.partial)
        self.assertEqual(
            result.text,
            "\n".join(("OPEN_APP: com.example", "Validate login screen is visible")),
        )
        self.assertEqual(result.review.commands[-1].source_steps, (1,))
        self.assertEqual(result.review.commands[-1].verified_by, ("completion_assertion",))
        self.assertEqual(result.review.lineage[-1].verified_by, ("completion_assertion",))

    def test_compose_keeps_one_terminal_validation_when_baseline_already_has_it(self) -> None:
        """
        Completed composed drafts do not duplicate a baseline terminal validation.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="search",
            goal="search",
            package="com.example",
            partial=False,
            steps=(
                EvidenceStep(index=1, event="action", action="tap"),
                EvidenceStep(index=2, event="validation", action="validate"),
            ),
        )
        validation = "Validate login screen is visible"
        baseline = AuthoringBaseline(
            content="\n".join(("OPEN_APP: com.example", validation)),
            partial=False,
            commands=(
                AuthoringBaselineCommand(text="OPEN_APP: com.example", source_steps=(1,)),
                AuthoringBaselineCommand(text=validation, source_steps=(2,)),
            ),
        )
        drafts = (self.__draft(step=1, text="OPEN_APP: com.example"),)

        result = composer.compose(
            drafts=drafts,
            evidence=evidence,
            baseline=baseline,
            completion=CompletionValidation(
                lines=(validation,),
                required=True,
                source_steps=(2,),
            ),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.text.splitlines().count(validation), 1)
        self.assertEqual(result.review.commands[-1].text, validation)
        self.assertEqual(result.review.commands[-1].source_steps, (2,))
        self.assertEqual(
            result.review.commands[-1].verified_by, ("execution", "completion_assertion")
        )

    def test_completion_replaces_terminal_state_runtime_validation(self) -> None:
        """
        Assertion-grounded terminal validation replaces terminal-state runtime validation.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="checkout",
            goal="cart verified",
            package="com.example",
            partial=False,
            steps=(
                EvidenceStep(
                    index=1,
                    event="action",
                    action="tap",
                    outcome=StepOutcome(changed=True),
                ),
                EvidenceStep(index=2, event="validation", action="validate"),
                EvidenceStep(index=3, event="action", action="store"),
            ),
        )
        baseline = AuthoringBaseline(
            content="\n".join(
                (
                    "Tap on View Cart",
                    "Validate cart status is visible",
                    "Store 186 as amount",
                )
            ),
            partial=False,
            commands=(
                AuthoringBaselineCommand(text="Tap on View Cart", source_steps=(1,)),
                AuthoringBaselineCommand(text="Validate cart status is visible", source_steps=(2,)),
                AuthoringBaselineCommand(text="Store 186 as amount", source_steps=(3,)),
            ),
        )

        result = composer.compose(
            drafts=(),
            evidence=evidence,
            baseline=baseline,
            completion=CompletionValidation(
                required=True,
                lines=("Validate cart contents and total are visible",),
                source_steps=(3,),
            ),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.text,
            "\n".join(
                (
                    "Tap on View Cart",
                    "Store 186 as amount",
                    "Validate cart contents and total are visible",
                )
            ),
        )

    def test_compose_marks_completed_draft_partial_when_terminal_validation_is_missing(
        self,
    ) -> None:
        """
        Completed composed drafts cannot claim completion when terminal validation is unavailable.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="search",
            goal="search",
            package="com.example",
            partial=False,
            steps=(EvidenceStep(index=1, event="action", action="tap"),),
        )
        drafts = (self.__draft(step=1, text="OPEN_APP: com.example"),)

        result = composer.compose(
            drafts=drafts,
            evidence=evidence,
            completion=CompletionValidation(required=True),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.review.partial)
        self.assertEqual(
            result.review.reason,
            "Completion assertions could not be rendered into a terminal validation.",
        )

    def test_partially_covered_merged_baseline_command_is_not_reused(self) -> None:
        """
        A baseline command cannot represent a step already covered by another command.
        """

        composer = StepDraftComposer()
        evidence = Evidence(
            intent="search",
            goal="search",
            package="com.example",
            partial=True,
            steps=(
                EvidenceStep(index=1, event="action", action="swipe_up"),
                EvidenceStep(index=2, event="action", action="swipe_up"),
            ),
        )
        baseline = AuthoringBaseline(
            content="\n".join(("Scroll down", 'Scroll down until "product is visible"')),
            partial=True,
            commands=(
                AuthoringBaselineCommand(text="Scroll down", source_steps=(1,)),
                AuthoringBaselineCommand(
                    text='Scroll down until "product is visible"',
                    source_steps=(1, 2),
                ),
            ),
        )
        drafts = (
            self.__draft(step=1, text="Scroll down"),
            self.__draft(step=2, text='Scroll down until "product card is visible"'),
        )

        result = composer.compose(drafts=drafts, evidence=evidence, baseline=baseline)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(
            result.text,
            "\n".join(("Scroll down", 'Scroll down until "product card is visible"')),
        )
        covered = [step for command in result.review.commands for step in command.source_steps]
        self.assertEqual(covered, [1, 2])

    @staticmethod
    def __draft(*, step: int, text: str) -> AuthoringDraft:
        """
        Build a generated step draft.
        """

        return AuthoringDraft(
            step_index=step,
            kind=AuthoringKind.STEP,
            status=AuthoringStatus.GENERATED,
            execution_id="execution-1",
            artifact=AuthoringArtifact(
                content=text,
                dialect=DialectName.DRIZZ,
                kind=AuthoringArtifactKind.TEXT,
            ),
        )
