"""
Unit tests for the completion gate family. Pin the per-kind evidence
contract so adding or removing a signal updates exactly one gate.
"""

from __future__ import annotations

import pytest

from fathom.core.agent.completion import (
    ActionStepGate,
    CompletionEvidence,
    CompletionGateFactory,
    ValidationStepGate,
)
from fathom.core.exceptions import ConfigurationError
from fathom.schemas.reasoning import SubGoalCompletionSignal
from fathom.schemas.subgoal import SubGoalKind


class TestActionStepGate:
    """
    Behavioral pins for :class:`ActionStepGate`.
    """

    @staticmethod
    def __signal(
        *,
        action_executed: bool = False,
        screen_verified: bool = False,
        flagged_complete: bool = False,
        rationale_verified: bool = False,
    ) -> SubGoalCompletionSignal:
        """
        Build a signal with the explicit evidence bits set.
        """

        return SubGoalCompletionSignal(
            action_executed=action_executed,
            screen_verified=screen_verified,
            flagged_complete=flagged_complete,
            rationale_verified=rationale_verified,
        )

    def test_complete_when_both_dimensions_present(self) -> None:
        """
        Both ``claim_verified`` and ``action_effective`` true → complete.
        """

        signal = self.__signal(
            action_executed=True,
            screen_verified=True,
            flagged_complete=True,
            rationale_verified=True,
        )
        verdict = ActionStepGate().evaluate(signal=signal)

        assert verdict.missing == []
        assert verdict.complete is True

    def test_missing_claim_blocks_completion(self) -> None:
        """
        Only ``action_effective`` set → blocked with claim missing.
        """

        signal = self.__signal(action_executed=True, screen_verified=True)
        verdict = ActionStepGate().evaluate(signal=signal)

        assert verdict.complete is False
        assert CompletionEvidence.CLAIM_VERIFIED in verdict.missing

    def test_missing_action_effective_blocks_completion(self) -> None:
        """
        Only ``claim_verified`` set → blocked with action missing.
        """

        signal = self.__signal(flagged_complete=True, rationale_verified=True)
        verdict = ActionStepGate().evaluate(signal=signal)

        assert verdict.complete is False
        assert CompletionEvidence.ACTION_EFFECTIVE in verdict.missing

    def test_no_evidence_lists_both_missing(self) -> None:
        """
        Empty signal → blocked with both dimensions missing.
        """

        verdict = ActionStepGate().evaluate(signal=self.__signal())

        assert verdict.complete is False
        assert set(verdict.missing) == {
            CompletionEvidence.CLAIM_VERIFIED,
            CompletionEvidence.ACTION_EFFECTIVE,
        }

    def test_partial_claim_does_not_satisfy(self) -> None:
        """
        ``flagged_complete`` alone (without rationale) must not satisfy ``claim_verified``.
        """

        signal = self.__signal(flagged_complete=True, action_executed=True, screen_verified=True)
        verdict = ActionStepGate().evaluate(signal=signal)

        assert verdict.complete is False
        assert CompletionEvidence.CLAIM_VERIFIED in verdict.missing


class TestValidationStepGate:
    """
    Behavioral pins for :class:`ValidationStepGate`.
    """

    def test_flagged_complete_alone_is_sufficient(self) -> None:
        """
        Validation steps require only the explicit completion flag; the screen is not expected to change.
        """

        signal = SubGoalCompletionSignal(flagged_complete=True)
        verdict = ValidationStepGate().evaluate(signal=signal)

        assert verdict.complete is True

    def test_missing_flagged_complete_blocks(self) -> None:
        """
        Without an explicit completion flag the validation step stalls.
        """

        signal = SubGoalCompletionSignal(rationale_verified=True, action_executed=True)
        verdict = ValidationStepGate().evaluate(signal=signal)

        assert verdict.complete is False
        assert CompletionEvidence.FLAGGED_COMPLETE in verdict.missing


class TestCompletionGateFactory:
    """
    Behavioral pins for :class:`CompletionGateFactory`.
    """

    def test_resolves_action_kind(self) -> None:
        """
        ACTION kind must yield an :class:`ActionStepGate`.
        """

        gate = CompletionGateFactory.for_kind(kind=SubGoalKind.ACTION)
        assert isinstance(gate, ActionStepGate)

    def test_resolves_validation_kind(self) -> None:
        """
        VALIDATION kind must yield a :class:`ValidationStepGate`.
        """

        gate = CompletionGateFactory.for_kind(kind=SubGoalKind.VALIDATION)
        assert isinstance(gate, ValidationStepGate)

    def test_unknown_kind_raises(self) -> None:
        """
        An unregistered kind must raise :class:`ConfigurationError`.
        """

        class _Bogus:
            value = "bogus"

        with pytest.raises(ConfigurationError):
            CompletionGateFactory.for_kind(kind=_Bogus())  # type: ignore[arg-type]
