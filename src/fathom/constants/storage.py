from __future__ import annotations

from enum import StrEnum


class StorageBackend(StrEnum):
    """
    Canonical identifiers for artifact storage backends.
    """

    LOCAL = "LOCAL"
    CLOUD = "CLOUD"
