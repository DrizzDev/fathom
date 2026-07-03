from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.flow import IssueCode
from fathom.constants.generation import ScriptSource, ScriptStatus
from fathom.schemas.flow import Issue
from fathom.schemas.generation import (
    GenerationFailure,
    GenerationResult,
    ScriptFileMetadata,
    ScriptReview,
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
