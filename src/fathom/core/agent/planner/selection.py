from __future__ import annotations

from fathom.core.agent.reasoner import Reasoner
from fathom.core.agent.state import AgentState
from fathom.core.exceptions import InvariantViolation
from fathom.schemas.actions import Action
from fathom.schemas.results import AnalysisResult


class ActionSelector:
    """
    Selects the best executable action from the analysis, avoiding this turn's known failed actions.
    """

    @staticmethod
    def select(*, state: AgentState, reasoner: Reasoner, analysis: AnalysisResult) -> Action:
        """
        Return the reasoner's best action, or raise when the analysis carries no executable action.
        """

        failures_raw = state.build_context().get("relevant_failures", [])
        failures = failures_raw if isinstance(failures_raw, list) else []

        if analysis.action is None:
            raise InvariantViolation("Planner action selection requires an executable action.")

        return reasoner.select_best_action(
            primary=analysis.action,
            alternatives=analysis.alternatives,
            failed_actions={str(failure) for failure in failures},
        )
