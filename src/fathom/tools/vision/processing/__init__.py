"""
Backward compatibility shim for processing module.

This module re-exports from the new processing/ location to maintain
backward compatibility during the hexagonal architecture migration.
"""

from __future__ import annotations

# Re-export from new location
from fathom.processing.annotator import ImageAnnotator
from fathom.processing.drawer import BoundsGenerator
from fathom.processing.geometry import GeometryUtils

__all__ = [
    "ImageAnnotator",
    "BoundsGenerator",
    "GeometryUtils",
]
