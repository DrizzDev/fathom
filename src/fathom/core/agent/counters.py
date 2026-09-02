"""
Run counters measured from what actually happened.

A benchmark host asks four questions about a finished run that only the harness can
answer: how many actions were dispatched, how many changed nothing, how many were
dispatched on coordinates nothing corroborated, and whether a human was involved. Those
answers used to be emitted as hardcoded zeros, which is worse than emitting nothing:
the host's fields are tri-state precisely so an uninstrumented harness reports
UNVERIFIED, and a fabricated zero turns a check nobody performed into one that passed.

Everything here is derived from recorded step results. Nothing is defaulted, and the
measurement is pure so it can be tested without a device or a provider.
"""

from __future__ import annotations

from typing import Optional, Sequence

from fathom.constants.state import CompletionReason
from fathom.core.capability.catalog import CommandCatalogProvider
from fathom.schemas.actions import CoordinateSource
from fathom.schemas.results import RunCounters
from fathom.schemas.steps import StepResult


def _dispatched(*, results: Sequence[StepResult]) -> list[StepResult]:
    """
    The steps that actually touched the device.

    A VALIDATE, a WAIT or a memory command executes without dispatching anything, so
    counting them would inflate the total and — because they never change the screen —
    would report every one of them as a no-op. The catalog owns that taxonomy already.
    """

    catalog = CommandCatalogProvider().build()
    dispatched: list[StepResult] = []

    for result in results:
        if not result.executed:
            continue
        action_type = result.step.action.action_type
        if catalog.is_control(action_type=action_type) or catalog.is_non_interactive(
            action_type=action_type
        ):
            continue
        dispatched.append(result)

    return dispatched


def measure_run(
    *,
    step_results: Sequence[StepResult],
    human_interventions: int,
    completion_reason: Optional[str],
) -> RunCounters:
    """
    Count what the recorded steps show, with no defaults anywhere.

    ``uncorroborated`` counts only ``CoordinateSource.MODEL`` — planner bounds dispatched
    on last-resort blind trust. ``MODEL_GROUNDED`` is deliberately excluded: those bounds
    came from the planner too, but OCR evidence inside the proposed region corroborated
    them, which is the whole distinction the cascade draws.
    """

    dispatched = _dispatched(results=step_results)

    uncorroborated = 0
    for result in dispatched:
        bounds = result.step.action.bounds
        if bounds is not None and bounds.source is CoordinateSource.MODEL:
            uncorroborated += 1

    return RunCounters(
        total_actions=len(dispatched),
        no_op_actions=sum(1 for result in dispatched if not result.screen_changed),
        uncorroborated_dispatches=uncorroborated,
        human_interventions=human_interventions,
        # The run stopped because it needed an answer and did not get one. Any other
        # ending means nothing was left outstanding, however the run turned out.
        awaiting_human_at_end=(completion_reason == CompletionReason.INTERVENTION_REQUIRED.value),
    )


__all__ = ["RunCounters", "measure_run"]
