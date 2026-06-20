from __future__ import annotations

from pypika.terms import LiteralValue, Term

from fathom.constants.storage import (
    INTERACTION_SQL_LIKE_ESCAPE_CHARACTER,
    SQL_LIKE_WILDCARD_PERCENT,
    SQL_LIKE_WILDCARD_UNDERSCORE,
)
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery


class SqlLikeEscape:
    """
    Neutralize SQL LIKE wildcards in user-supplied substrings.
    """

    @classmethod
    def escape(cls, *, value: str) -> str:
        """
        Return value with backslash, percent, and underscore escaped for LIKE.
        """

        return (
            value.replace(
                INTERACTION_SQL_LIKE_ESCAPE_CHARACTER,
                INTERACTION_SQL_LIKE_ESCAPE_CHARACTER * 2,
            )
            .replace(
                SQL_LIKE_WILDCARD_PERCENT,
                f"{INTERACTION_SQL_LIKE_ESCAPE_CHARACTER}{SQL_LIKE_WILDCARD_PERCENT}",
            )
            .replace(
                SQL_LIKE_WILDCARD_UNDERSCORE,
                f"{INTERACTION_SQL_LIKE_ESCAPE_CHARACTER}{SQL_LIKE_WILDCARD_UNDERSCORE}",
            )
        )

    @classmethod
    def prefix_clause(
        cls,
        *,
        column: Term,
        prefix: str,
        binder: ParameterizedQuery,
    ) -> LiteralValue:
        """
        Build a `LIKE ? ESCAPE` predicate that matches an escaped prefix value.
        """

        bound = f"{cls.escape(value=prefix)}{SQL_LIKE_WILDCARD_PERCENT}"

        placeholder = binder.bind_placeholder(value=bound)
        rendered = (
            f"{column.get_sql(quote_char=None)} LIKE {placeholder} "
            f"ESCAPE '{INTERACTION_SQL_LIKE_ESCAPE_CHARACTER}'"
        )

        return LiteralValue(rendered)
