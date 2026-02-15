"""
Backward compatibility shim for processing.parsers module.

This module re-exports from the new processing/parsers/ location to maintain
backward compatibility during the hexagonal architecture migration.
"""

from __future__ import annotations

# Re-export from new location
from fathom.processing.parsers.android import AndroidParser
from fathom.processing.parsers.base import PlatformParser
from fathom.processing.parsers.factory import PlatformParserFactory

__all__ = [
    "AndroidParser",
    "PlatformParser",
    "PlatformParserFactory",
]
