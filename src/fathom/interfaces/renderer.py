from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.flow import Flow


class Renderer(ABC):
    """
    Port that renders a target-neutral flow into dialect script text.
    """

    @abstractmethod
    def render(self, *, flow: Flow) -> str:
        """
        Render the flow into dialect-specific script text.
        """

        raise NotImplementedError
