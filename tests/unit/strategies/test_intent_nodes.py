from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Optional, cast

import pytest

from fathom.constants.execution import PLANNER_RETRY_ESCALATION_THRESHOLD
from fathom.constants.state import CompletionReason
from fathom.schemas.results import PlanResult
from fathom.strategies.graph.intent.nodes import IntentNodeProvider


def _launcher_persistence_decision(
    provider: IntentNodeProvider, *, execution_activity: str, observed_activity: str
) -> bool:
    """Call the private launcher-persistence helper in a mypy-safe way."""

    decision_function = cast(
        "Callable[..., bool]",
        provider.__getattribute__("_IntentNodeProvider__should_skip_launcher_persistence"),
    )
    return decision_function(
        execution_activity=execution_activity,
        observed_activity=observed_activity,
    )


def test_should_skip_launcher_persistence_on_launcher_only_steps() -> None:
    """Skip persistence when execution both starts and ends on launcher packages."""

    provider = object.__new__(IntentNodeProvider)

    should_skip = _launcher_persistence_decision(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="com.google.android.apps.nexuslauncher",
        provider=provider,
    )

    assert should_skip is True


def test_should_persist_launcher_to_app_transition() -> None:
    """Persist steps that leave the launcher and open the requested app."""

    provider = object.__new__(IntentNodeProvider)

    should_skip = _launcher_persistence_decision(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="com.snabbit.customer",
        provider=provider,
    )

    assert should_skip is False


def test_should_skip_when_observed_package_is_unknown() -> None:
    """Keep launcher suppression when post-action package resolution is unavailable."""

    provider = object.__new__(IntentNodeProvider)

    should_skip = _launcher_persistence_decision(
        execution_activity="com.google.android.apps.nexuslauncher",
        observed_activity="unknown",
        provider=provider,
    )

    assert should_skip is True


# ---------------------------------------------------------------------------
# __maybe_escalate_on_planner_retries
# ---------------------------------------------------------------------------


def _make_provider_with_retry_context(
    *,
    sub_goal_planner_retries: int,
    replan_return_value: Optional[dict[str, Any]],
) -> tuple[IntentNodeProvider, list[dict[str, Any]]]:
    """Build an IntentNodeProvider stub with just enough context to exercise
    the replan-escalation helper in isolation.

    Stubs out ``__context.agent_state.sub_goal_planner_retries`` and
    ``__replan_remaining_sub_goals`` (via name-mangled attribute
    assignment) so the helper can be driven without the full graph.
    Returns the provider and a list into which any
    ``__replan_remaining_sub_goals`` invocations are recorded.
    """

    provider = object.__new__(IntentNodeProvider)
    provider._IntentNodeProvider__context = SimpleNamespace(  # type: ignore[attr-defined]
        agent_state=SimpleNamespace(sub_goal_planner_retries=sub_goal_planner_retries),
    )

    recorded_calls: list[dict[str, Any]] = []

    async def _fake_replan(*, capture: Any, failure_reason: str) -> Any:
        recorded_calls.append({"capture": capture, "failure_reason": failure_reason})
        return replan_return_value

    provider._IntentNodeProvider__replan_remaining_sub_goals = _fake_replan  # type: ignore[attr-defined]
    return provider, recorded_calls


async def _call_maybe_escalate(
    provider: IntentNodeProvider,
    *,
    plan: PlanResult,
    capture: Any,
) -> Optional[dict[str, Any]]:
    helper = cast(
        "Callable[..., Awaitable[Optional[dict[str, Any]]]]",
        provider.__getattribute__("_IntentNodeProvider__maybe_escalate_on_planner_retries"),
    )
    return await helper(plan=plan, capture=capture)


def _make_blocked_plan() -> PlanResult:
    return PlanResult(
        rationale=CompletionReason.ACTION_BLOCKED.value,
        is_complete=False,
        should_retry=True,
        step=None,
    )


@pytest.mark.asyncio
async def test_maybe_escalate_fires_when_counter_at_threshold() -> None:
    """When the planner-retry counter >= threshold and plan rationale is
    ACTION_BLOCKED, the helper must call __replan_remaining_sub_goals and
    return its result."""

    fake_replan_result = {"IS_COMPLETE": False, "SHOULD_RETRY": True}
    provider, recorded_calls = _make_provider_with_retry_context(
        sub_goal_planner_retries=PLANNER_RETRY_ESCALATION_THRESHOLD,
        replan_return_value=fake_replan_result,
    )

    capture_sentinel = object()
    result = await _call_maybe_escalate(
        provider, plan=_make_blocked_plan(), capture=capture_sentinel
    )

    assert result is fake_replan_result
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["capture"] is capture_sentinel
    assert "3 times" in recorded_calls[0]["failure_reason"]


@pytest.mark.asyncio
async def test_maybe_escalate_no_op_below_threshold() -> None:
    """Counter below threshold: helper must return None without calling replan."""

    provider, recorded_calls = _make_provider_with_retry_context(
        sub_goal_planner_retries=PLANNER_RETRY_ESCALATION_THRESHOLD - 1,
        replan_return_value={"IS_COMPLETE": False, "SHOULD_RETRY": True},
    )

    result = await _call_maybe_escalate(provider, plan=_make_blocked_plan(), capture=object())

    assert result is None
    assert recorded_calls == []


@pytest.mark.asyncio
async def test_maybe_escalate_no_op_when_rationale_not_blocked() -> None:
    """Even at threshold, non-ACTION_BLOCKED rationale must not escalate."""

    provider, recorded_calls = _make_provider_with_retry_context(
        sub_goal_planner_retries=PLANNER_RETRY_ESCALATION_THRESHOLD,
        replan_return_value={"IS_COMPLETE": False, "SHOULD_RETRY": True},
    )

    unrelated_plan = PlanResult(
        rationale="Some other reason",
        is_complete=False,
        should_retry=False,
        step=None,
    )
    result = await _call_maybe_escalate(provider, plan=unrelated_plan, capture=object())

    assert result is None
    assert recorded_calls == []


@pytest.mark.asyncio
async def test_maybe_escalate_propagates_none_from_failed_replan() -> None:
    """When the decomposer replan returns None (exception caught inside), the
    helper must also return None so the caller falls through to normal retry."""

    provider, recorded_calls = _make_provider_with_retry_context(
        sub_goal_planner_retries=PLANNER_RETRY_ESCALATION_THRESHOLD + 1,
        replan_return_value=None,
    )

    result = await _call_maybe_escalate(provider, plan=_make_blocked_plan(), capture=object())

    assert result is None
    assert len(recorded_calls) == 1


# ---------------------------------------------------------------------------
# __advance_after_verify_passed
# ---------------------------------------------------------------------------


class _RecordingTelemetry:
    """Telemetry stub that appends every ``info`` call into ``events``."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def info(self, message: str, **context: Any) -> None:
        self.events.append({"message": message, **context})

    async def debug(self, *_args: Any, **_kwargs: Any) -> None: ...
    async def warning(self, *_args: Any, **_kwargs: Any) -> None: ...
    async def error(self, *_args: Any, **_kwargs: Any) -> None: ...
    async def exception(self, *_args: Any, **_kwargs: Any) -> None: ...


def _build_advance_provider(
    *,
    sub_goal_descriptions: list[str],
) -> tuple[IntentNodeProvider, _RecordingTelemetry, Any]:
    """Build a bare IntentNodeProvider wired to a real AgentState + recording telemetry."""

    from fathom.core.agent.state import AgentState
    from fathom.schemas.subgoal import SubGoal, SubGoalStatus

    agent_state = AgentState(intent="test-intent", max_steps=10)
    agent_state.set_sub_goals(
        [
            SubGoal(
                index=index,
                description=description,
                status=SubGoalStatus.PENDING,
                confidence=0.9,
            )
            for index, description in enumerate(sub_goal_descriptions)
        ]
    )

    telemetry = _RecordingTelemetry()
    context_manager = SimpleNamespace(clear_user_guidance=lambda: None)

    provider = object.__new__(IntentNodeProvider)
    provider._IntentNodeProvider__context = SimpleNamespace(  # type: ignore[attr-defined]
        agent_state=agent_state,
        telemetry=telemetry,
        context_manager=context_manager,
    )
    return provider, telemetry, agent_state


async def _call_advance(
    provider: IntentNodeProvider,
    *,
    is_subgoal_verify: bool,
    current_sub_goal: Any,
    reason: str,
) -> Any:
    helper = cast(
        "Callable[..., Awaitable[Any]]",
        provider.__getattribute__("_IntentNodeProvider__advance_after_verify_passed"),
    )
    return await helper(
        is_subgoal_verify=is_subgoal_verify,
        current_sub_goal=current_sub_goal,
        reason=reason,
    )


@pytest.mark.asyncio
async def test_advance_emits_completed_then_started_when_next_sub_goal_exists() -> None:
    """Verifying sub-goal 0 of 2 must emit SUB_GOAL_COMPLETED for goal 0
    then SUB_GOAL_STARTED for goal 1, in that order, with total=2."""

    from fathom.constants import FathomEvent
    from fathom.constants.state import CommonStateKey, IntentStateKey

    provider, telemetry, agent_state = _build_advance_provider(
        sub_goal_descriptions=["open app", "tap login"],
    )
    current_sub_goal = agent_state.get_current_sub_goal()

    result = await _call_advance(
        provider,
        is_subgoal_verify=True,
        current_sub_goal=current_sub_goal,
        reason="sub-goal met per screenshot",
    )

    # Two cues in order: COMPLETED(idx=0), STARTED(idx=1).
    cue_events = [
        event
        for event in telemetry.events
        if event.get("type")
        in {
            FathomEvent.SUB_GOAL_COMPLETED,
            FathomEvent.SUB_GOAL_STARTED,
        }
    ]
    assert [event["type"] for event in cue_events] == [
        FathomEvent.SUB_GOAL_COMPLETED,
        FathomEvent.SUB_GOAL_STARTED,
    ]
    assert cue_events[0]["index"] == 0
    assert cue_events[0]["total"] == 2
    assert cue_events[0]["description"] == "open app"
    assert cue_events[1]["index"] == 1
    assert cue_events[1]["total"] == 2
    assert cue_events[1]["description"] == "tap login"

    # Result is a retry, not terminal.
    assert result[CommonStateKey.IS_COMPLETE] is False
    assert result[IntentStateKey.SHOULD_RETRY] is True


@pytest.mark.asyncio
async def test_advance_completes_intent_when_last_sub_goal_passes() -> None:
    """Verifying the only (last) sub-goal must emit SUB_GOAL_COMPLETED
    exactly once, produce no SUB_GOAL_STARTED, and return a terminal
    ``is_complete=True`` result."""

    from fathom.constants import FathomEvent
    from fathom.constants.state import CommonStateKey

    provider, telemetry, agent_state = _build_advance_provider(
        sub_goal_descriptions=["single goal"],
    )
    current_sub_goal = agent_state.get_current_sub_goal()

    result = await _call_advance(
        provider,
        is_subgoal_verify=True,
        current_sub_goal=current_sub_goal,
        reason="goal met",
    )

    completed = [
        event for event in telemetry.events if event.get("type") == FathomEvent.SUB_GOAL_COMPLETED
    ]
    started = [
        event for event in telemetry.events if event.get("type") == FathomEvent.SUB_GOAL_STARTED
    ]

    assert len(completed) == 1
    assert completed[0]["index"] == 0
    assert completed[0]["total"] == 1
    assert started == []

    assert result[CommonStateKey.IS_COMPLETE] is True
    assert result[CommonStateKey.COMPLETION_REASON] == (
        "All sub-goals completed and verified sequentially"
    )
