"""
Pins for :class:`EscapeReport` and :class:`EscapeCategory`.

The escape primitive is the agent's typed signal that it cannot make
safe forward progress on the active sub-goal. The category drives
routing — replan against the current screen vs escalate to the human
— so the typed contract must be enforced at construction time
(engineering standards §17 Boundary Validation, §19 Fail Fast).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.schemas.escape import (
    HUMAN_ESCAPE_CATEGORIES,
    REPLAN_ESCAPE_CATEGORIES,
    EscapeCategory,
    EscapeReport,
)


class TestEscapeCategoryRouting:
    """
    Pins the routing partition between replan and human categories.
    """

    def test_replan_and_human_partitions_are_disjoint(self) -> None:
        """
        No category may belong to both partitions, otherwise routing
        becomes ambiguous.
        """

        assert REPLAN_ESCAPE_CATEGORIES.isdisjoint(HUMAN_ESCAPE_CATEGORIES)

    def test_every_category_has_a_routing_decision(self) -> None:
        """
        Adding an :class:`EscapeCategory` value without classifying it
        into one of the routing partitions is a structural bug — the
        planner would have no branch for the new category.
        """

        partitioned = REPLAN_ESCAPE_CATEGORIES | HUMAN_ESCAPE_CATEGORIES
        unclassified = set(EscapeCategory) - partitioned
        assert unclassified == set(), (
            f"Unclassified EscapeCategory values: {sorted(c.value for c in unclassified)}"
        )


class TestEscapeReport:
    """
    Behavioural pins for the typed escape payload.
    """

    def test_replan_category_routes_to_replan(self) -> None:
        """
        Replan-partition categories must report ``routes_to_replan``.
        """

        report = EscapeReport(
            category=EscapeCategory.TARGET_NOT_AVAILABLE,
            detail="no Continue button on this screen",
        )
        assert report.routes_to_replan() is True
        assert report.routes_to_human() is False

    def test_human_category_routes_to_human(self) -> None:
        """
        Human-partition categories must report ``routes_to_human``.
        """

        report = EscapeReport(
            category=EscapeCategory.UNSAFE_ACTION,
            detail="tapping Delete account would be irreversible",
        )
        assert report.routes_to_human() is True
        assert report.routes_to_replan() is False

    def test_empty_detail_is_rejected(self) -> None:
        """
        An empty detail defeats the typed contract — the decomposer
        preamble or HITL prompt would carry no justification. Pydantic
        must reject this at construction.
        """

        with pytest.raises(ValidationError):
            EscapeReport(category=EscapeCategory.WRONG_SCREEN, detail="")

    def test_report_is_frozen(self) -> None:
        """
        Reports are value objects; mutation after construction must
        raise.
        """

        report = EscapeReport(
            category=EscapeCategory.WRONG_SCREEN,
            detail="on the debug overlay, not the home screen",
        )
        with pytest.raises(ValidationError):
            report.detail = "different"  # type: ignore[misc]
