"""
Backward compatibility shim for annotator module.

This module re-exports from the new processing/ location to maintain
backward compatibility during the hexagonal architecture migration.
"""

from __future__ import annotations

# Re-export from new location
from fathom.processing.annotator import ImageAnnotator

__all__ = ["ImageAnnotator"]
