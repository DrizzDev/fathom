from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def sanitize_path_component(value: str) -> str:
    """Remove characters that are unsafe for filenames/path segments."""
    return "".join(char for char in value if char.isalnum() or char in "._-")


@dataclass(frozen=True)
class StorageMetadata:
    """Validated and sanitized storage metadata extracted from a raw dict."""

    session: str
    package: str
    activity: str
    category: str
    filename: str | None  # None means the caller should generate a default name


def extract_metadata(raw: Dict[str, Any]) -> StorageMetadata:
    """Validate required fields and return sanitized StorageMetadata.

    Raises ValueError when required fields (session_id, package_name) are missing.
    """
    if not raw:
        raise ValueError("Storage metadata is required")

    session = raw.get("session_id")
    package = raw.get("package_name")
    activity = raw.get("activity_name") or package
    category = raw.get("category", "screenshot")
    filename = raw.get("filename")

    if not all([session, package]):
        raise ValueError(f"Missing required storage metadata: {session=}, {package=}")

    return StorageMetadata(
        session=sanitize_path_component(str(session)),
        package=sanitize_path_component(str(package)),
        activity=sanitize_path_component(str(activity)),
        category=sanitize_path_component(str(category)),
        filename=filename,
    )
