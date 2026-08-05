from __future__ import annotations

from typing import Optional

from fathom.schemas.success import CommandSuccess, ObservationRequirement, ObservedSuccess, Success


class Eligibility:
    """
    The single authority for which observation a goal's completion must be proven visually, if any.
    """

    @staticmethod
    def observation(*, success: Success) -> Optional[ObservationRequirement]:
        """
        Return the observation a visual assessment must prove for this goal, or None when it proves from a receipt.
        """

        if isinstance(success, ObservedSuccess):
            return success.observation
        if isinstance(success, CommandSuccess) and success.postcondition is not None:
            return success.postcondition
        return None
