from __future__ import annotations

from fathom.adapters.dialect.drizz.parser import DrizzLarkParser
from fathom.core.dialect.drizz.dialect import Dialect as DrizzDialect
from fathom.interfaces.dialect import Dialect as DialectPort


class DrizzDialectFactory:
    """
    Assembles the fully wired Drizz dialect: renderer, Lark parser, and round-trip checker.
    """

    def create(self) -> DialectPort:
        """
        Build the Drizz dialect bound to its Lark parser.
        """

        return DrizzDialect(parser=DrizzLarkParser())
