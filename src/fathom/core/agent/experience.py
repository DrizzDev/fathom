from __future__ import annotations

from fathom.schemas.experience import Experience


class ExperiencePolicy:
    """
    Decides which past experiences may surface to the planner as positive hints.
    """

    def positive(self, *, experience: Experience) -> bool:
        """
        Return whether the experience proves the action truly moved its task forward.
        """

        return experience.advanced
