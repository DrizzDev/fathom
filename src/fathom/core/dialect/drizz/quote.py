from __future__ import annotations

from typing import FrozenSet, Tuple

from fathom.constants.dialect.drizz import Phrase, Quote, State


class Quoting:
    """
    Wraps a value in a parser-safe Drizz string delimiter, preferring double quotes.
    """

    __ESCAPE = "\\"
    __LINE_BREAKS: FrozenSet[str] = frozenset({"\n", "\r", "\t"})
    __ORDER: Tuple[Quote, ...] = (Quote.DOUBLE, Quote.SINGLE, Quote.BACKTICK)

    # Literals that terminate a free-text tail (capture -> "as", nl_target -> "until", subject -> STATE);
    # only these, embedded in recorded text, would be mis-lexed and break a bare round-trip.
    __RESERVED_PHRASES: Tuple[str, ...] = tuple(state.value for state in State)
    __RESERVED_WORDS: FrozenSet[str] = frozenset({Phrase.AS.value, Phrase.UNTIL.value})

    # Characters the unquoted FREEWORD terminal excludes; their presence forces a quoted form.
    __DELIMITERS: FrozenSet[str] = frozenset(
        f"{Quote.DOUBLE.value}{Quote.SINGLE.value}{Quote.BACKTICK.value};{{}}"
    )

    def wrap(self, *, value: str) -> str:
        """
        Return the value delimited and escaped so it remains one Drizz string token.
        """

        quote = self.__quote(value=value)
        escaped = self.__escape(value=value, quote=quote)

        return f"{quote}{escaped}{quote}"

    def __quote(self, *, value: str) -> Quote:
        """
        Return the preferred delimiter, choosing double quotes when every delimiter appears.
        """

        for quote in self.__ORDER:
            if str(quote) not in value:
                return quote

        return Quote.DOUBLE

    def __escape(self, *, value: str, quote: Quote) -> str:
        """
        Escape the active delimiter, backslashes, and line-breaking characters.
        """

        out = value.replace(self.__ESCAPE, f"{self.__ESCAPE}{self.__ESCAPE}")
        out = out.replace(str(quote), f"{self.__ESCAPE}{quote}")
        out = out.replace("\n", f"{self.__ESCAPE}n")
        out = out.replace("\r", f"{self.__ESCAPE}r")

        return out.replace("\t", f"{self.__ESCAPE}t")

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

        if self.__ESCAPE in value or any(char in value for char in self.__LINE_BREAKS):
            return True

        lowered = value.lower()
        if any(word in self.__RESERVED_WORDS for word in lowered.split()):
            return True

        return any(phrase in lowered for phrase in self.__RESERVED_PHRASES)
