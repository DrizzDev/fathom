"""
Backward compatibility shim for drawer module.

This module re-exports from the new processing/ location to maintain
backward compatibility during the hexagonal architecture migration.
"""

from __future__ import annotations

# Re-export from new location
from fathom.processing.drawer import BoundsGenerator

__all__ = ["BoundsGenerator"]
