from __future__ import annotations

from fathom.constants.dialect import DialectName
from fathom.core.dialect.drizz.check import Checker
from fathom.core.dialect.drizz.print import CanonicalPrinter
from fathom.core.dialect.drizz.render import Renderer
from fathom.interfaces.checker import Checker as CheckPort
from fathom.interfaces.dialect import Dialect as DialectBase
from fathom.interfaces.parser import DrizzParser
from fathom.interfaces.renderer import Renderer as RenderPort


class Dialect(DialectBase):
    """
    The Drizz dialect binding its renderer and checker.
    """

    def __init__(self, *, parser: DrizzParser) -> None:
        """
        Bind the Drizz renderer and a checker driven by the injected parser.
        """

        self.__renderer: RenderPort = Renderer()
        self.__checker: CheckPort = Checker(parser=parser, printer=CanonicalPrinter())

    @property
    def name(self) -> DialectName:
        """
        Identifier of the Drizz dialect.
        """

        return DialectName.DRIZZ

    @property
    def renderer(self) -> RenderPort:
        """
        Renderer that converts a flow into Drizz text.
        """

        return self.__renderer

    @property
    def checker(self) -> CheckPort:
        """
        Checker that validates Drizz text.
        """

        return self.__checker
