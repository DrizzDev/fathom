from __future__ import annotations

import unittest

from pypika import Table
from pypika.functions import Coalesce, Lower

from fathom.constants.storage import SqlParameterStyle
from fathom.infrastructure.interaction.pypika.escaping import SqlLikeEscape
from fathom.infrastructure.interaction.pypika.query import ParameterizedQuery


class TestSqlLikeEscapeEscape(unittest.TestCase):
    """
    SqlLikeEscape.escape neutralizes wildcards and the escape character itself.
    """

    def test_percent_becomes_escaped(self) -> None:
        """
        A bare percent must be prefixed with a single backslash.
        """

        self.assertEqual(SqlLikeEscape.escape(value="a%b"), "a\\%b")

    def test_underscore_becomes_escaped(self) -> None:
        """
        A bare underscore must be prefixed with a single backslash.
        """

        self.assertEqual(SqlLikeEscape.escape(value="a_b"), "a\\_b")

    def test_backslash_is_doubled_before_wildcard_escaping(self) -> None:
        """
        A backslash in the input must be doubled so it remains literal.
        """

        self.assertEqual(SqlLikeEscape.escape(value="a\\b"), "a\\\\b")

    def test_plain_value_is_unchanged(self) -> None:
        """
        Values without wildcards or backslashes must pass through verbatim.
        """

        self.assertEqual(SqlLikeEscape.escape(value="abc"), "abc")

    def test_backslash_and_wildcard_combination(self) -> None:
        """
        A backslash followed by a wildcard must not collapse into a fake escape.
        """

        self.assertEqual(SqlLikeEscape.escape(value="\\%"), "\\\\\\%")


class TestSqlLikeEscapePrefixClause(unittest.TestCase):
    """
    SqlLikeEscape.prefix_clause emits a LIKE/ESCAPE predicate over a pypika column.
    """

    def setUp(self) -> None:
        """
        Build a question-mark binder and a `lower(coalesce(title, ''))` column term.
        """

        self.__threads: Table = Table("threads")
        self.__binder: ParameterizedQuery = ParameterizedQuery(
            parameter_style=SqlParameterStyle.QUESTION_MARK,
        )

    def test_clause_renders_like_with_escape_keyword(self) -> None:
        """
        The rendered SQL must contain `LIKE ? ESCAPE '\\'` after the column term.
        """

        clause = SqlLikeEscape.prefix_clause(
            column=Lower(Coalesce(self.__threads.title, "")),
            prefix="acme",
            binder=self.__binder,
        )
        rendered = clause.get_sql(quote_char=None)

        self.assertIn("LIKE ?", rendered)
        self.assertTrue(rendered.endswith("ESCAPE '\\'"))

    def test_clause_appends_wildcard_to_escaped_prefix(self) -> None:
        """
        The bound parameter must contain the escaped prefix followed by `%`.
        """

        SqlLikeEscape.prefix_clause(
            column=Lower(Coalesce(self.__threads.title, "")),
            prefix="a%b",
            binder=self.__binder,
        )

        self.assertEqual(self.__binder.parameters, ("a\\%b%",))

    def test_clause_uses_numbered_placeholder_under_numbered_style(self) -> None:
        """
        Under NUMBERED style the placeholder must be `$N` not `?`.
        """

        numbered_binder = ParameterizedQuery(parameter_style=SqlParameterStyle.NUMBERED)
        clause = SqlLikeEscape.prefix_clause(
            column=Lower(Coalesce(self.__threads.title, "")),
            prefix="acme",
            binder=numbered_binder,
        )

        self.assertIn("LIKE $1", clause.get_sql(quote_char=None))
