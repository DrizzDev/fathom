from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.dialect.drizz.script import DrizzScript


class DrizzParser(ABC):
    """
    Port that parses Drizz script text into a typed command AST.
    """

    @abstractmethod
    def parse(self, *, text: str) -> DrizzScript:
        """
        Parse Drizz text into a typed DrizzScript, raising LanguageParseError on failure.
        """

        raise NotImplementedError
