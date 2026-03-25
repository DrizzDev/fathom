import xml.etree.ElementTree as ET  # nosec
from typing import List, Type

from fathom.core.exceptions import ConfigurationError
from fathom.processing.parsers.android import AndroidParser
from fathom.processing.parsers.base import PlatformParser
from fathom.processing.parsers.ios import IOSParser


class PlatformParserFactory:
    """
    Factory class for creating platform parsers.
    """

    __parsers: List[Type[PlatformParser]] = [IOSParser, AndroidParser]

    @classmethod
    def get_parser(cls, root: ET.Element) -> PlatformParser:
        """
        Get a parser for the platform.
        """

        for parser in cls.__parsers:
            if parser.is_platform_match(root):
                return parser()

        raise ConfigurationError(
            f"Unable to resolve platform parser for XML root tag '{root.tag}'."
        )
