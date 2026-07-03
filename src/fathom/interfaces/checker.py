from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.flow import Report


class Checker(ABC):
    """
    Port that validates rendered dialect script text and reports any issues.
    """

    @abstractmethod
    def check(self, *, text: str) -> Report:
        """
        Validate the rendered script text and return any issues.
        """

        raise NotImplementedError
