"""
Pins for :class:`AgentState`'s action-effect trajectory plumbing.

The Phase 1A signal contract introduced a rolling window of
:class:`ActionEffect` records on :class:`AgentState`. These tests
exercise the public surface used by:

- The ANALYZE prompt assembler (renders the trajectory window).
- The RECORD node's stuck-recovery dispatcher (reads
  ``consecutive_no_progress_count`` to decide whether to fire the
  ``NO_PROGRESS`` recovery trigger).
"""

from __future__ import annotations

from fathom.constants.screen import NO_PROGRESS_RECOVERY_THRESHOLD
from fathom.core.agent.state import AgentState
from fathom.schemas.effect import ActionEffect, ActionEffectStatus


def _effect(status: ActionEffectStatus, *, visual_progress: float = 0.0) -> ActionEffect:
    """
    Build a minimal :class:`ActionEffect` for trajectory tests.
    """

    return ActionEffect(
        status=status,
        visual_progress=visual_progress,
        phash_distance=0,
    )


class TestActionEffectTrajectory:
    """
    Behavioural pins for :class:`AgentState` action-effect tracking.
    """

    def test_recent_effects_empty_before_any_record(self) -> None:
        """
        A fresh :class:`AgentState` has no recorded effects.
        """

        state = AgentState(intent="x")
        assert state.get_recent_effects() == []
        assert state.get_last_action_effect() is None
        assert state.consecutive_no_progress_count == 0

    def test_record_action_effect_round_trips_through_accessor(self) -> None:
        """
        Recording an effect makes it observable through both the
        rolling-window accessor and the most-recent accessor.
        """

        state = AgentState(intent="x")
        effect = _effect(ActionEffectStatus.PROGRESS, visual_progress=0.42)
        state.record_action_effect(effect=effect)

        assert state.get_recent_effects() == [effect]
        assert state.get_last_action_effect() == effect

    def test_consecutive_no_progress_counts_only_trailing_tail(self) -> None:
        """
        A trailing ``PROGRESS`` resets the counter — the tail is the
        contiguous run of ``NO_PROGRESS`` at the end of the window.
        """

        state = AgentState(intent="x")
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))
        state.record_action_effect(effect=_effect(ActionEffectStatus.PROGRESS))
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))

        assert state.consecutive_no_progress_count == 1

    def test_consecutive_no_progress_counts_recovery_threshold_in_a_row(self) -> None:
        """
        :data:`NO_PROGRESS_RECOVERY_THRESHOLD` consecutive
        ``NO_PROGRESS`` classifications cross the recovery escalation
        bar — RECORD dispatches ``NO_PROGRESS`` when this is met.
        """

        state = AgentState(intent="x")
        for _ in range(NO_PROGRESS_RECOVERY_THRESHOLD):
            state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))

        assert state.consecutive_no_progress_count == NO_PROGRESS_RECOVERY_THRESHOLD

    def test_uncertain_breaks_no_progress_run(self) -> None:
        """
        ``UNCERTAIN`` is not ``NO_PROGRESS`` — it breaks the trailing
        tail count. The agent gets a free try on ambiguous outcomes
        rather than being escalated immediately.
        """

        state = AgentState(intent="x")
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))
        state.record_action_effect(effect=_effect(ActionEffectStatus.UNCERTAIN))
        state.record_action_effect(effect=_effect(ActionEffectStatus.NO_PROGRESS))

        assert state.consecutive_no_progress_count == 1
