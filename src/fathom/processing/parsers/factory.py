import xml.etree.ElementTree as ET  # nosec
from typing import List, Optional, Type

from fathom.processing.parsers.android import AndroidParser
from fathom.processing.parsers.base import PlatformParser


class PlatformParserFactory:
    """
    Factory class for creating platform parsers.
    """

    __parsers: List[Type[PlatformParser]] = [AndroidParser]

    @classmethod
    def get_parser(cls, root: ET.Element) -> Optional[PlatformParser]:
        """
        Get a parser for the platform.
        """

        for parser in cls.__parsers:
            if parser.is_platform_match(root):
                return parser()

        return AndroidParser()
