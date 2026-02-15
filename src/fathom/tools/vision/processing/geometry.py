"""
Backward compatibility shim for geometry module.

This module re-exports from the new processing/ location to maintain
backward compatibility during the hexagonal architecture migration.
"""

from __future__ import annotations

# Re-export from new location
from fathom.processing.geometry import GeometryUtils

__all__ = ["GeometryUtils"]
