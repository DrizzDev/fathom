from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fathom.constants import ActionType
from fathom.core.runtime import RuntimeState
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.healing import HealingDecision, HealingDecisionKind
from fathom.schemas.localization import LocalizationResult, LocalizationStatus
from fathom.schemas.observation import KeyboardObservation, ScreenObservation
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.steps import Step
from fathom.schemas.subgoal import SubGoal
from fathom.schemas.supervision import BlockReason, SupervisionVerdict, VerdictKind
from fathom.strategies.graph.intent.nodes.gate import ActionGate


def _action(*, target: str = "Continue") -> Action:
    """
    Plain TAP action used as the planned-step seed and as the healing
    alternative.
    """

    return Action(
        action_type=ActionType.TAP,
        target=target,
        rationale="t",
        confidence=1.0,
    )


def _step(*, action: Action) -> Step:
    """
    Step wrapping the supplied action; step number is arbitrary.
    """

    return Step(
        action=action,
        event_type="action",
        condition="x",
        screen_hash="0" * 16,
        step_number=1,
    )


def _capture() -> ScreenCapture:
    """
    Minimal screen capture; the gate consults the perception layer via
    the localizer, which is stubbed, so the bytes are placeholders.
    """

    return ScreenCapture(
        width=1000,
        height=2000,
        activity="app",
        image=b"PNG",
        timestamp=0,
    )


def _observation() -> ScreenObservation:
    """
    Minimal observation passed through to the healing request and to
    the (stubbed) localizer / supervisor.
    """

    return ScreenObservation(
        activity="app",
        elements=(),
        hashes=ScreenHashBundle(
            visual_hash="0" * 16,
            xml_hash="a" * 16,
            interaction_hash="b" * 16,
        ),
        keyboard=KeyboardObservation(visible=False),
    )


def _verdict(
    *,
    kind: VerdictKind,
    reason: BlockReason | None = None,
) -> SupervisionVerdict:
    """
    Supervision verdict fixture parameterised on kind and reason so
    each branch (blocked-with-reason, blocked-without-reason, allowed
    after healing) can be driven independently.
    """

    return SupervisionVerdict(kind=kind, reason=reason, message="m")


def _localization(
    *, status: LocalizationStatus = LocalizationStatus.RESOLVED
) -> LocalizationResult:
    """
    Localization result fixture. Resolved (with bounds) is the only
    state that lets ``apply_localization`` attach bounds back to the
    step; other states pass through unchanged.
    """

    return LocalizationResult(
        status=status,
        point=None,
        bounds=Bounds(x=10, y=10, width=20, height=20, coordinate_system="pixel"),
        confidence=1.0,
    )


class _StubContext:
    """
    :class:`GraphContext` test double exposing only the surface
    :meth:`ActionGate.heal_blocked_action` actually consumes.

    Carries a real :class:`RuntimeState` so the per-task healing
    counter increments are observable on the live aggregate. Every
    collaborator (healing orchestrator, resolution, localizer,
    supervisor) is a configurable async / sync mock.
    """

    def __init__(
        self,
        *,
        decision: HealingDecision,
        substitute_action: Action,
        post_heal_verdict: SupervisionVerdict,
        localization: LocalizationResult,
    ) -> None:
        """
        Pre-seed every collaborator with the response that drives the
        branch under test.
        """

        sub_goals = [SubGoal(index=0, description="Open the app")]
        runtime = RuntimeState.create()
        self.agent_state = SimpleNamespace(
            runtime=runtime,
            get_current_sub_goal=lambda: sub_goals[0],
        )
        self.healing_orchestrator = SimpleNamespace(decide=AsyncMock(return_value=decision))
        self.resolution = SimpleNamespace(
            substitute=AsyncMock(return_value=substitute_action),
        )
        self.target_localizer = SimpleNamespace(
            localize=AsyncMock(return_value=localization),
        )
        self.runtime_supervisor = SimpleNamespace(
            supervise=MagicMock(return_value=post_heal_verdict),
        )
        self.event_emitter = SimpleNamespace(emit=AsyncMock())
        self.workflow_id = "run-test"


def _persistence() -> Any:
    """
    Persistence stub used only by the unrelated ``blocked_execute_result``
    code path; ``heal_blocked_action`` does not touch it but the gate
    constructor requires one.
    """

    return SimpleNamespace(persist=MagicMock())


class ActionGateHealBlockedActionTest(unittest.IsolatedAsyncioTestCase):
    """
    Pins :meth:`ActionGate.heal_blocked_action`.

    Drives the three real branches: verdict-without-reason short-circuit,
    healing returns NOT_APPLICABLE (kind != TRY_ACTION), and healing
    returns TRY_ACTION → step substituted → re-supervised → either
    ALLOW (healed step returned) or BLOCK (None returned).
    """

    async def test_verdict_without_reason_returns_none(self) -> None:
        """
        A verdict with no ``reason`` cannot drive healing; the method
        returns ``None`` without invoking the orchestrator. This guards
        against malformed verdicts surfacing as healed retries.
        """

        decision = HealingDecision(
            kind=HealingDecisionKind.TRY_ACTION,
            action=_action(target="alt"),
            reason="t",
        )
        ctx = _StubContext(
            decision=decision,
            substitute_action=_action(target="alt"),
            post_heal_verdict=_verdict(kind=VerdictKind.ALLOW),
            localization=_localization(),
        )
        gate = ActionGate(context=ctx, persistence=_persistence())  # type: ignore[arg-type]

        result = await gate.heal_blocked_action(
            step=_step(action=_action()),
            capture=_capture(),
            verdict=_verdict(kind=VerdictKind.BLOCK, reason=None),
            observation=_observation(),
        )

        self.assertIsNone(result)
        ctx.healing_orchestrator.decide.assert_not_called()

    async def test_non_try_action_decision_returns_none(self) -> None:
        """
        Healing returning anything other than TRY_ACTION (FAIL_BOUNDED,
        REQUEST_REPLAN, etc.) yields ``None`` — the gate cannot use
        non-action decisions to retry execution.
        """

        decision = HealingDecision(
            kind=HealingDecisionKind.FAIL_BOUNDED,
            action=None,
            reason="bounded failure",
        )
        ctx = _StubContext(
            decision=decision,
            substitute_action=_action(),
            post_heal_verdict=_verdict(kind=VerdictKind.ALLOW),
            localization=_localization(),
        )
        gate = ActionGate(context=ctx, persistence=_persistence())  # type: ignore[arg-type]

        result = await gate.heal_blocked_action(
            step=_step(action=_action()),
            capture=_capture(),
            verdict=_verdict(kind=VerdictKind.BLOCK, reason=BlockReason.REPEATED_NO_EFFECT),
            observation=_observation(),
        )

        self.assertIsNone(result)

    async def test_try_action_allowed_after_re_supervision_returns_healed_step(self) -> None:
        """
        TRY_ACTION with the healed step re-supervised as ALLOW returns
        the substituted step. The per-task healing counter must tick
        exactly once so the budget accounting is correct.
        """

        decision = HealingDecision(
            kind=HealingDecisionKind.TRY_ACTION,
            action=_action(target="alt"),
            reason="try alternative",
        )
        ctx = _StubContext(
            decision=decision,
            substitute_action=_action(target="alt"),
            post_heal_verdict=_verdict(kind=VerdictKind.ALLOW),
            localization=_localization(),
        )
        gate = ActionGate(context=ctx, persistence=_persistence())  # type: ignore[arg-type]

        healed = await gate.heal_blocked_action(
            step=_step(action=_action()),
            capture=_capture(),
            verdict=_verdict(kind=VerdictKind.BLOCK, reason=BlockReason.REPEATED_NO_EFFECT),
            observation=_observation(),
        )

        self.assertIsNotNone(healed)
        assert healed is not None
        self.assertEqual(healed.action.target, "alt")
        # The runtime healing counter must tick exactly once per
        # heal_blocked_action call (budget accounting invariant).
        active_task = gate.active_execution_task()
        self.assertEqual(
            ctx.agent_state.runtime.healing.task_count(task_id=active_task.identifier),
            1,
        )

    async def test_try_action_blocked_on_re_supervision_returns_none(self) -> None:
        """
        TRY_ACTION whose healed step still blocks on re-supervision
        returns ``None`` — the gate does not retry recursively, it
        surfaces the block to the caller.
        """

        decision = HealingDecision(
            kind=HealingDecisionKind.TRY_ACTION,
            action=_action(target="alt"),
            reason="try alternative",
        )
        ctx = _StubContext(
            decision=decision,
            substitute_action=_action(target="alt"),
            post_heal_verdict=_verdict(
                kind=VerdictKind.BLOCK,
                reason=BlockReason.UNSAFE_ACTION,
            ),
            localization=_localization(),
        )
        gate = ActionGate(context=ctx, persistence=_persistence())  # type: ignore[arg-type]

        result = await gate.heal_blocked_action(
            step=_step(action=_action()),
            capture=_capture(),
            verdict=_verdict(kind=VerdictKind.BLOCK, reason=BlockReason.REPEATED_NO_EFFECT),
            observation=_observation(),
        )

        self.assertIsNone(result)
