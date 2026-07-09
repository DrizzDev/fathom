from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.constants.dialect import DialectName
from fathom.interfaces.checker import Checker
from fathom.interfaces.renderer import Renderer


class Dialect(ABC):
    """
    Port binding the renderer and checker for one script dialect.
    """

    @property
    @abstractmethod
    def name(self) -> DialectName:
        """
        Identifier of this dialect.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def renderer(self) -> Renderer:
        """
        Renderer that converts a flow into this dialect's text.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def checker(self) -> Checker:
        """
        Checker that validates this dialect's rendered text.
        """

        raise NotImplementedError
