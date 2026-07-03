from __future__ import annotations

from typing import cast

from lark import Lark
from lark.exceptions import LarkError

from fathom.adapters.dialect.drizz.grammar import DRIZZ_GRAMMAR
from fathom.adapters.dialect.drizz.transform import DrizzTransformer
from fathom.core.exceptions import LanguageParseError
from fathom.interfaces.parser import DrizzParser
from fathom.schemas.dialect.drizz.script import DrizzScript


class DrizzLarkParser(DrizzParser):
    """
    Parses rendered Drizz text into a typed DrizzScript via a Lark LALR grammar.
    """

    def __init__(self) -> None:
        """
        Compile the grammar and bind the AST transformer.
        """

        self.__transformer = DrizzTransformer()
        self.__parser = Lark(DRIZZ_GRAMMAR, parser="lalr", start="start")

    def parse(self, *, text: str) -> DrizzScript:
        """
        Parse Drizz text into a typed DrizzScript, raising LanguageParseError on failure.
        """

        try:
            tree = self.__parser.parse(text)
            return cast("DrizzScript", self.__transformer.transform(tree))
        except LarkError as exception:
            raise LanguageParseError(str(exception)) from exception
