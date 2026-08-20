from __future__ import annotations

from typing import Optional

from fathom.core.agent.state import AgentState
from fathom.schemas.escalation import StuckSource


class StuckSourceResolver:
    """
    Resolve the canonical "stuck" source that warrants escalation.

    Pure read-only over :class:`AgentState`; returns ``None`` when no source is active so callers
    skip the gate. Priority is deterministic — budget exhaustion (a hard signal) beats the loop
    detector (a probabilistic sliding-window classifier). Global ``max_steps`` is not modelled
    here: the analyze node already terminates the workflow when it is hit.
    """

    def resolve(self, *, agent_state: AgentState) -> Optional[StuckSource]:
        """
        Return the highest-priority source, or ``None`` when no source applies.
        """

        if agent_state.current_sub_goal_over_budget:
            return StuckSource.SUBGOAL_BUDGET

        if agent_state.is_stuck:
            return StuckSource.LOOP_DETECTOR

        return None
