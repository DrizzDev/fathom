from __future__ import annotations

from fathom.constants import NEXT_PHASE_ACTION_TYPES, ActionType


class OpenerSignalPolicy:
    """
    Detects whether a planned command signals an opener sub-goal has advanced past its opening phase.
    """

    def advanced(self, *, action_type: ActionType) -> bool:
        """
        Return whether this command, planned during an opener sub-goal, signals the next phase began.
        """

        return action_type in NEXT_PHASE_ACTION_TYPES
