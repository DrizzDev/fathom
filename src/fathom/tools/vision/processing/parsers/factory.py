"""
Backward compatibility shim for parsers.factory module.

This module re-exports from the new processing/parsers/ location to maintain
backward compatibility during the hexagonal architecture migration.
"""

from __future__ import annotations

# Re-export from new location
from fathom.processing.parsers.factory import PlatformParserFactory

__all__ = ["PlatformParserFactory"]
