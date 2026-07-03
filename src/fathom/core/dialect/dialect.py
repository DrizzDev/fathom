from __future__ import annotations

from typing import Dict

from fathom.constants.dialect import DialectName
from fathom.core.exceptions import ConfigurationError
from fathom.interfaces.dialect import Dialect


class DialectRegistry:
    """
    Resolves a dialect implementation by name; the seam for adding new dialects.
    """

    def __init__(self) -> None:
        """
        Initialise an empty dialect registry.
        """

        self.__dialects: Dict[DialectName, Dialect] = {}

    def register(self, *, dialect: Dialect) -> None:
        """
        Register a dialect implementation under its name.
        """

        self.__dialects[dialect.name] = dialect

    def resolve(self, *, name: DialectName) -> Dialect:
        """
        Return the registered dialect for the given name.
        """

        dialect = self.__dialects.get(name)

        if dialect is None:
            raise ConfigurationError(f"No dialect registered for '{name}'.")

        return dialect
