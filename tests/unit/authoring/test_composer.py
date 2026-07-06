from __future__ import annotations

import unittest
from typing import Tuple

from fathom.authoring.application.composer import StepDraftComposer
from fathom.constants.authoring import AuthoringArtifactKind, AuthoringKind, AuthoringStatus
from fathom.constants.dialect import DialectName
from fathom.constants.flow import AssertionSource, CheckKind
from fathom.constants.generation import ScriptSource
from fathom.schemas.authoring import AuthoringArtifact, AuthoringBaseline, AuthoringBaselineCommand
from fathom.schemas.authoring.draft import AuthoringDraft
from fathom.schemas.flow import CompletionAssertion, Evidence, EvidenceStep


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
            completion_validation=("Validate login screen is visible",),
            require_completion_validation=True,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.review.partial)
        self.assertEqual(
            result.text,
            "\n".join(("OPEN_APP: com.example", "Validate login screen is visible")),
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
            drafts=drafts, evidence=evidence, require_completion_validation=True
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.review.partial)
        self.assertEqual(
            result.review.reason,
            "Completion assertions could not be rendered into a terminal validation.",
        )

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
