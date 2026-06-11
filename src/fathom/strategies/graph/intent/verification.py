from __future__ import annotations

from fathom.constants.state import IntentStateKey, VerifyMode
from fathom.core.agent.state import AgentState
from fathom.core.exceptions import InvariantViolation
from fathom.strategies.graph.state import IntentGraphState


class VerificationModePolicy:
    """
    Selects the verifier mode for graph producers and VERIFY consumers.
    """

    def mode_for_producer(self, *, agent_state: AgentState) -> VerifyMode:
        """
        Return the mode a producer should stamp before routing to VERIFY.
        """

        current = agent_state.get_current_sub_goal()
        if current is None or agent_state.all_sub_goals_complete():
            return VerifyMode.FULL_INTENT

        if agent_state.has_active_final_sub_goal():
            return VerifyMode.PENDING_FINAL_COMMIT

        return VerifyMode.SUB_GOAL

    def mode_for_verify(
        self,
        *,
        state: IntentGraphState,
        agent_state: AgentState,
    ) -> VerifyMode:
        """
        Read VERIFY_MODE from graph state, falling back to legacy inference.
        """

        declared = state.get(IntentStateKey.VERIFY_MODE)
        if declared is None:
            return self.mode_for_producer(agent_state=agent_state)

        if not isinstance(declared, str):
            raise InvariantViolation(
                f"VERIFY_MODE graph state must be a string; got {type(declared).__name__}."
            )

        try:
            return VerifyMode(declared)
        except ValueError as exception:
            raise InvariantViolation(
                f"VERIFY_MODE has unrecognized value {declared!r}; "
                "likely a corrupted checkpoint or schema drift."
            ) from exception
