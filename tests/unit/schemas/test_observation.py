from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.schemas.observation import LoopObservation, ScreenRelation


class TestLoopObservation:
    """
    Schema-shape invariants for the structured loop observation.
    """

    def test_minimal_observation_constructs(self) -> None:
        """
        A minimal observation requires only the repeated action, a
        count >= 2, and a :class:`ScreenRelation`. Other fields default.
        """

        obs = LoopObservation(
            repeated_action="Swipe up on Auto suggest page",
            count=3,
            screen_relation=ScreenRelation.NEAR_DUPLICATE,
        )

        assert obs.count == 3
        assert obs.note is None
        assert obs.progress_scores == []
        assert obs.suggested_alternatives == []
        assert obs.screen_relation == ScreenRelation.NEAR_DUPLICATE

    def test_count_below_two_is_rejected(self) -> None:
        """
        ``count`` is bounded ``ge=2`` — a single occurrence is not a
        loop and must not be expressible as a LoopObservation.
        """

        with pytest.raises(ValidationError):
            LoopObservation(
                repeated_action="x",
                count=1,
                screen_relation=ScreenRelation.NEAR_DUPLICATE,
            )

    def test_screen_relation_enum_values_are_stable(self) -> None:
        """
        Pin the StrEnum values consumed by the prompt template.
        """

        assert ScreenRelation.DIVERGING.value == "diverging"
        assert ScreenRelation.NEAR_DUPLICATE.value == "near_duplicate"
        assert ScreenRelation.OSCILLATING.value == "oscillating"

    def test_observation_is_frozen(self) -> None:
        """
        :class:`LoopObservation` is a value object — assignments after
        construction must be rejected.
        """

        obs = LoopObservation(
            repeated_action="x",
            count=2,
            screen_relation=ScreenRelation.DIVERGING,
        )

        with pytest.raises(ValidationError):
            obs.count = 99  # type: ignore[misc]
