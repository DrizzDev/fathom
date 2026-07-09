from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.flow import IssueCode
from fathom.constants.generation import ScriptSource, ScriptStatus
from fathom.schemas.flow import Issue
from fathom.schemas.generation import (
    CompletionValidation,
    GenerationFailure,
    GenerationResult,
    ScriptCommand,
    ScriptFileMetadata,
    ScriptReview,
    ScrollCollapseState,
)


class ScriptFileMetadataTest(unittest.TestCase):
    """
    Pins backward-compatible loading and the script source/status/review contract.
    """

    def test_defaults_to_generated_quality_with_clean_review(self) -> None:
        """
        A metadata with only a review defaults to a generated quality artifact.
        """

        metadata = ScriptFileMetadata(review=ScriptReview(partial=False))

        self.assertIs(metadata.status, ScriptStatus.GENERATED)
        self.assertIs(metadata.source, ScriptSource.QUALITY)
        self.assertFalse(metadata.review.partial)
        self.assertEqual(metadata.review.discarded, ())

    def test_failed_baseline_metadata_round_trips(self) -> None:
        """
        A failed-baseline sidecar preserves source, status, issues, and review across serialization.
        """

        issue = Issue(code=IssueCode.UNGROUNDED_STORE, message="no matching capture")
        metadata = ScriptFileMetadata(
            status=ScriptStatus.FAILED,
            source=ScriptSource.BASELINE,
            issues=(issue,),
            review=ScriptReview(partial=True, reason="no validation"),
        )

        restored = ScriptFileMetadata.model_validate_json(metadata.model_dump_json())

        self.assertIs(restored.status, ScriptStatus.FAILED)
        self.assertIs(restored.source, ScriptSource.BASELINE)
        self.assertEqual(restored.issues[0].code, IssueCode.UNGROUNDED_STORE)
        self.assertTrue(restored.review.partial)
        self.assertEqual(restored.review.reason, "no validation")


class GenerationResultTest(unittest.TestCase):
    """
    Pins that a successful generation can never carry empty script text.
    """

    def test_empty_text_is_rejected(self) -> None:
        """
        A GenerationResult must carry non-empty rendered script text.
        """

        with self.assertRaises(ValidationError):
            GenerationResult(text="", attempts=1)

    def test_review_defaults_when_omitted(self) -> None:
        """
        A complete, non-partial run defaults to a clean review.
        """

        result = GenerationResult(text="OPEN_APP:com.example", attempts=1)

        self.assertFalse(result.review.partial)
        self.assertEqual(result.review.discarded, ())


class CompletionValidationTest(unittest.TestCase):
    """
    Pins terminal completion validation semantics for fallback composition.
    """

    def test_optional_completion_validation_is_not_missing(self) -> None:
        """
        Optional completion validation is satisfied without rendered lines.
        """

        validation = CompletionValidation()

        self.assertFalse(validation.missing)

    def test_required_completion_validation_without_lines_is_missing(self) -> None:
        """
        Required completion validation is missing until rendered lines exist.
        """

        validation = CompletionValidation(required=True)

        self.assertTrue(validation.missing)

    def test_required_completion_validation_with_lines_is_satisfied(self) -> None:
        """
        Required completion validation is satisfied by rendered validation lines.
        """

        validation = CompletionValidation(
            required=True,
            lines=("Validate login screen is visible",),
        )

        self.assertFalse(validation.missing)


class ScriptCommandTest(unittest.TestCase):
    """
    Pins rendered command provenance defaults.
    """

    def test_command_defaults_to_unverified_non_screen_authored_text(self) -> None:
        """
        Rendered commands do not claim verification unless a producer supplies it.
        """

        command = ScriptCommand(text="Tap on product card", source_steps=(2,))

        self.assertEqual(command.verified_by, ())
        self.assertFalse(command.screen_authored)


class ScrollCollapseStateTest(unittest.TestCase):
    """
    Pins repeated-command collapse state without coupling it to line positions.
    """

    def test_repeats_only_when_command_and_region_match(self) -> None:
        """
        A command repeats only inside the same active recovery region.
        """

        state = ScrollCollapseState(command="scroll", region=1)

        self.assertTrue(state.repeats(command="scroll", region=1))
        self.assertFalse(state.repeats(command="scroll", region=2))
        self.assertFalse(state.repeats(command="tap", region=1))

    def test_advance_resets_when_command_or_region_is_absent(self) -> None:
        """
        Missing command or region resets the collapse state.
        """

        state = ScrollCollapseState(command="scroll", region=1)

        self.assertEqual(state.advance(command=None, region=1), ScrollCollapseState())
        self.assertEqual(state.advance(command="scroll", region=None), ScrollCollapseState())


class GenerationFailureTest(unittest.TestCase):
    """
    Pins that a failure explains itself and is distinct from an empty success.
    """

    def test_failure_requires_at_least_one_issue(self) -> None:
        """
        A GenerationFailure must carry at least one blocking issue.
        """

        with self.assertRaises(ValidationError):
            GenerationFailure(issues=())

    def test_failure_carries_issues_and_review(self) -> None:
        """
        A failure exposes its blocking issues and review context.
        """

        issue = Issue(code=IssueCode.SYNTAX_ERROR, message="not canonical")
        failure = GenerationFailure(
            issues=(issue,), review=ScriptReview(partial=True, reason="no validation")
        )

        self.assertEqual(failure.issues[0].code, IssueCode.SYNTAX_ERROR)
        self.assertTrue(failure.review.partial)
