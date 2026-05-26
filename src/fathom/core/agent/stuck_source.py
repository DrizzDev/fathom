from __future__ import annotations

from logging import getLogger
from typing import Optional

from fathom.core.agent.state import AgentState
from fathom.schemas.escalation import StuckSource

logger = getLogger(__name__)


class StuckSourceResolver:
    """
    Resolve the canonical "stuck" source that warrants escalation.

    Pure read-only over :class:`AgentState`. Returns ``None`` when no source is
    active so callers can short-circuit without invoking the gate. Priority
    order is deterministic: budget exhaustion beats loop detection because
    budget is an unambiguous hard signal whereas the loop detector is a
    probabilistic classifier over a sliding window.

    Global ``max_steps`` is not modelled here — the analyze node already
    terminates the workflow when it is hit, so a source for it would be
    unreachable and would also overlap a code path with different semantics.
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
