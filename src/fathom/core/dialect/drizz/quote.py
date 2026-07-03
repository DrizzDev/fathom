from __future__ import annotations

from typing import FrozenSet, Tuple

from fathom.constants.dialect.drizz import Phrase, Quote, State
from fathom.core.exceptions import LanguageComplianceError


class Quoting:
    """
    Wraps a value in the first Drizz delimiter it does not contain, preferring double quotes.
    """

    __ORDER: Tuple[Quote, ...] = (Quote.DOUBLE, Quote.SINGLE, Quote.BACKTICK)

    # Literals that terminate a free-text tail (capture -> "as", nl_target -> "until", subject -> STATE);
    # only these, embedded in recorded text, would be mis-lexed and break a bare round-trip.
    __RESERVED_PHRASES: Tuple[str, ...] = tuple(state.value for state in State)
    __RESERVED_WORDS: FrozenSet[str] = frozenset({Phrase.AS.value, Phrase.UNTIL.value})

    # Characters the unquoted FREEWORD terminal excludes; their presence forces a quoted form.
    __DELIMITERS: FrozenSet[str] = frozenset('";{}')

    def wrap(self, *, value: str) -> str:
        """
        Return the value delimited by a non-colliding quote, failing if every quote collides.
        """

        for quote in self.__ORDER:
            if str(quote) not in value:
                return f"{quote}{value}{quote}"

        raise LanguageComplianceError(
            f"Value contains every Drizz quote type and cannot be rendered: {value!r}"
        )

    def conditional(self, *, value: str) -> str:
        """
        Quote a free-text phrase only when bare text would not round-trip; leave clean phrases unquoted.
        """

        return self.wrap(value=value) if self.__collides(value=value) else value

    def __collides(self, *, value: str) -> bool:
        """
        Whether the bare value would not round-trip: irregular whitespace or a reserved Drizz token.
        """

        if value != " ".join(value.split()):
            return True

        if any(char in value for char in self.__DELIMITERS):
            return True

        lowered = value.lower()
        if any(word in self.__RESERVED_WORDS for word in lowered.split()):
            return True

        return any(phrase in lowered for phrase in self.__RESERVED_PHRASES)
