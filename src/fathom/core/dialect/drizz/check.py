from __future__ import annotations

from itertools import zip_longest

from fathom.constants.flow import IssueCode
from fathom.core.dialect.drizz.print import CanonicalPrinter
from fathom.core.exceptions import LanguageParseError
from fathom.interfaces.checker import Checker as CheckPort
from fathom.interfaces.parser import DrizzParser
from fathom.schemas.flow import Issue, Report


class Checker(CheckPort):
    """
    Validates rendered Drizz by parsing it and confirming it round-trips to canonical text.
    """

    def __init__(self, *, parser: DrizzParser, printer: CanonicalPrinter) -> None:
        """
        Bind the grammar parser and the canonical printer.
        """

        self.__parser = parser
        self.__printer = printer

    def check(self, *, text: str) -> Report:
        """
        Validate the rendered script text and return any issues.
        """

        try:
            script = self.__parser.parse(text=text)
        except LanguageParseError as exception:
            return Report(issues=(Issue(code=IssueCode.SYNTAX_ERROR, message=str(exception)),))

        canonical = self.__normalise(text=self.__printer.emit(script=script))
        rendered = self.__normalise(text=text)

        if canonical != rendered:
            return Report(
                issues=(
                    Issue(
                        code=IssueCode.ROUND_TRIP_MISMATCH,
                        message=self.__mismatch(rendered=rendered, canonical=canonical),
                    ),
                )
            )

        return Report()

    def __mismatch(self, *, rendered: str, canonical: str) -> str:
        """
        Describe the first line where the rendered text diverges from canonical Drizz.
        """

        pairs = zip_longest(rendered.splitlines(), canonical.splitlines())
        for number, (got, want) in enumerate(pairs, start=1):
            if got != want:
                return (
                    "Rendered text is not canonical Drizz at line "
                    f"{number}: expected {want!r}, got {got!r}."
                )

        return "Rendered text is not canonical Drizz."

    def __normalise(self, *, text: str) -> str:
        """
        Strip trailing whitespace and surrounding blank lines for comparison.
        """

        return "\n".join(line.rstrip() for line in text.strip().splitlines())
