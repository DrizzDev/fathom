from __future__ import annotations

import unittest

from pydantic import ValidationError

from fathom.constants.localization import LocalizationGridScale
from fathom.schemas.actions import Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.localization import (
    LayoutMatchConfiguration,
    LocalizationProposal,
    MemberOutcome,
    PhraseMatch,
    ProposalCollection,
    VisionLocalizationPayload,
)


class VisionLocalizationPayloadTest(unittest.TestCase):
    """
    Pins the constrained-decoding contract the vision-localizer adapter relies on.
    """

    @staticmethod
    def __valid(**overrides: object) -> dict[str, object]:
        """
        Build a baseline payload mapping callers can override per assertion.
        """

        baseline: dict[str, object] = {
            "x1": 100,
            "y1": 200,
            "x2": 400,
            "y2": 500,
            "confidence": 0.9,
            "rationale": "Tight match on the visible button glyphs.",
        }
        baseline.update(overrides)
        return baseline

    def test_valid_payload_constructs(self) -> None:
        """
        A payload inside the grid bounds with positive area constructs cleanly.
        """

        payload = VisionLocalizationPayload(**self.__valid())

        self.assertEqual(payload.x1, 100)
        self.assertEqual(payload.y2, 500)
        self.assertEqual(payload.confidence, 0.9)
        self.assertFalse(payload.refused)

    def test_edge_above_grid_maximum_is_rejected(self) -> None:
        """
        Edges past the grid maximum violate the published contract.
        """

        with self.assertRaises(ValidationError):
            VisionLocalizationPayload(**self.__valid(x2=LocalizationGridScale.MAXIMUM + 1))

    def test_inverted_axis_payload_is_rejected(self) -> None:
        """
        Right edge at or before the left edge produces no usable rectangle.
        """

        with self.assertRaises(ValidationError):
            VisionLocalizationPayload(**self.__valid(x1=400, x2=400))

    def test_zero_area_non_refusal_payload_is_rejected(self) -> None:
        """
        A degenerate rectangle that is not the refusal sentinel must fail validation.
        """

        with self.assertRaises(ValidationError):
            VisionLocalizationPayload(**self.__valid(x1=500, x2=500, y1=600, y2=600))

    def test_all_zero_payload_is_refusal(self) -> None:
        """
        The canonical refusal sentinel constructs cleanly and reports refused.
        """

        refusal = VisionLocalizationPayload(
            x1=0,
            y1=0,
            x2=0,
            y2=0,
            confidence=0.0,
            rationale="Target not visible.",
        )

        self.assertTrue(refusal.refused)

    def test_confidence_above_unit_interval_is_rejected(self) -> None:
        """
        Confidence outside the closed unit interval must not reach the ensemble.
        """

        with self.assertRaises(ValidationError):
            VisionLocalizationPayload(**self.__valid(confidence=1.5))


class PhraseMatchTest(unittest.TestCase):
    """
    Pins the schema used by the layout localizer to publish merged phrase matches.
    """

    @staticmethod
    def __bounds() -> Bounds:
        """
        Build a representative pixel bounds rectangle for a clustered phrase.
        """

        return Bounds(
            x=1040,
            y=703,
            width=355,
            height=45,
            source=CoordinateSource.OCR,
            coordinate_system=CoordinateSystem.DEVICE_PIXEL,
        )

    def test_phrase_match_constructs_with_valid_inputs(self) -> None:
        """
        A valid phrase match captures text, bounds, score, and source token count.
        """

        match = PhraseMatch(
            text="Free Offers",
            bounds=self.__bounds(),
            score=0.95,
            confidence=0.92,
            token_count=2,
        )

        self.assertEqual(match.text, "Free Offers")
        self.assertEqual(match.token_count, 2)

    def test_score_above_unit_interval_is_rejected(self) -> None:
        """
        Match score must stay inside the closed unit interval.
        """

        with self.assertRaises(ValidationError):
            PhraseMatch(
                text="Free Offers",
                bounds=self.__bounds(),
                score=1.5,
                confidence=0.9,
                token_count=2,
            )

    def test_zero_token_count_is_rejected(self) -> None:
        """
        A phrase must be built from at least one token.
        """

        with self.assertRaises(ValidationError):
            PhraseMatch(
                text="Free Offers",
                bounds=self.__bounds(),
                score=0.9,
                confidence=0.9,
                token_count=0,
            )


class LayoutMatchConfigurationTest(unittest.TestCase):
    """
    Pins the tunables that govern phrase clustering and target matching.
    """

    def test_defaults_match_published_constants(self) -> None:
        """
        Defaults align with the named thresholds in constants/localization.py.
        """

        configuration = LayoutMatchConfiguration()

        self.assertEqual(configuration.phrase_match_threshold, 0.8)
        self.assertEqual(configuration.per_word_similarity_threshold, 0.8)
        self.assertEqual(configuration.min_word_length_for_fuzz, 3)
        self.assertEqual(configuration.min_token_confidence, 0.5)

    def test_threshold_above_unit_interval_is_rejected(self) -> None:
        """
        Phrase-match threshold must remain inside the closed unit interval.
        """

        with self.assertRaises(ValidationError):
            LayoutMatchConfiguration(phrase_match_threshold=1.5)

    def test_min_word_length_must_be_positive(self) -> None:
        """
        Short-word exact-match gate requires a strictly positive minimum length.
        """

        with self.assertRaises(ValidationError):
            LayoutMatchConfiguration(min_word_length_for_fuzz=0)


class MemberOutcomeTest(unittest.TestCase):
    """
    Pins MemberOutcome, migrated from a frozen dataclass to a Pydantic value object.
    """

    @staticmethod
    def __proposal() -> LocalizationProposal:
        """
        Build a representative member proposal.
        """

        return LocalizationProposal(
            bounds=Bounds(
                x=10,
                y=20,
                width=30,
                height=40,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            confidence=0.9,
            source="gemini_vision",
        )

    def test_successful_outcome_carries_proposal_and_defaults_to_not_failed(self) -> None:
        """
        A successful outcome holds the proposal and defaults failed to False.
        """

        proposal = self.__proposal()
        outcome = MemberOutcome(proposal=proposal)

        self.assertIs(outcome.proposal, proposal)
        self.assertFalse(outcome.failed)

    def test_none_proposal_with_failure_flag(self) -> None:
        """
        A failed member reports no proposal and a set failed flag.
        """

        outcome = MemberOutcome(proposal=None, failed=True)

        self.assertIsNone(outcome.proposal)
        self.assertTrue(outcome.failed)

    def test_equality_by_value(self) -> None:
        """
        Two outcomes with equal fields compare equal.
        """

        self.assertEqual(
            MemberOutcome(proposal=None, failed=True), MemberOutcome(proposal=None, failed=True)
        )

    def test_is_immutable(self) -> None:
        """
        A member outcome is frozen and rejects mutation.
        """

        outcome = MemberOutcome(proposal=None)

        with self.assertRaises(ValidationError):
            outcome.failed = True

    def test_rejects_invalid_proposal_type(self) -> None:
        """
        A non-proposal value for the proposal field fails validation.
        """

        with self.assertRaises(ValidationError):
            MemberOutcome.model_validate({"proposal": "not a proposal"})

    def test_rejects_unknown_field(self) -> None:
        """
        An unexpected field is rejected by the forbid-extra contract.
        """

        with self.assertRaises(ValidationError):
            MemberOutcome.model_validate({"proposal": None, "unexpected": 1})


class ProposalCollectionTest(unittest.TestCase):
    """
    Pins ProposalCollection, migrated from a frozen dataclass to a Pydantic value object.
    """

    @staticmethod
    def __proposal() -> LocalizationProposal:
        """
        Build a representative member proposal.
        """

        return LocalizationProposal(
            bounds=Bounds(
                x=10,
                y=20,
                width=30,
                height=40,
                source=CoordinateSource.MODEL,
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
            ),
            confidence=0.9,
            source="gemini_vision",
        )

    def test_holds_proposals_and_defaults_to_zero_failures(self) -> None:
        """
        A collection holds its proposals and defaults failed_members to zero.
        """

        proposals = [self.__proposal()]
        collection = ProposalCollection(proposals=proposals)

        self.assertEqual(collection.proposals, proposals)
        self.assertEqual(collection.failed_members, 0)

    def test_counts_failed_members(self) -> None:
        """
        A collection preserves the failed-member count.
        """

        self.assertEqual(ProposalCollection(proposals=[], failed_members=2).failed_members, 2)

    def test_proposals_remain_a_list(self) -> None:
        """
        The proposals field remains a list, not a tuple.
        """

        collection = ProposalCollection(proposals=[self.__proposal()])

        self.assertIsInstance(collection.proposals, list)

    def test_equality_by_value(self) -> None:
        """
        Two collections with equal fields compare equal.
        """

        self.assertEqual(
            ProposalCollection(proposals=[], failed_members=1),
            ProposalCollection(proposals=[], failed_members=1),
        )

    def test_is_immutable(self) -> None:
        """
        A proposal collection is frozen and rejects mutation.
        """

        collection = ProposalCollection(proposals=[])

        with self.assertRaises(ValidationError):
            collection.failed_members = 3

    def test_rejects_invalid_failed_members_type(self) -> None:
        """
        A non-integer failed-member count fails validation.
        """

        with self.assertRaises(ValidationError):
            ProposalCollection.model_validate({"proposals": [], "failed_members": "two"})

    def test_rejects_unknown_field(self) -> None:
        """
        An unexpected field is rejected by the forbid-extra contract.
        """

        with self.assertRaises(ValidationError):
            ProposalCollection.model_validate({"proposals": [], "unexpected": 1})
