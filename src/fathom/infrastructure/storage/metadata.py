from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


def sanitize_path_component(value: str) -> str:
    """
    Remove characters that are unsafe for filenames/path segments.
    """

    sanitized = "".join(char for char in value if char.isalnum() or char in "._-")
    # Reject empty or directory traversal segments like "." and ".."
    if sanitized in {"", ".", ".."}:
        raise ValueError(f"Invalid path component: {value!r}")
    return sanitized


class StorageMetadata(BaseModel):
    """
    Validated and sanitized storage metadata extracted from a raw dict.
    """

    model_config = ConfigDict(frozen=True)

    session: str = Field(description="Sanitized session identifier.")
    package: str = Field(description="Sanitized package name.")
    activity: str = Field(description="Sanitized activity identifier.")
    category: str = Field(description="Sanitized storage category.")
    filename: str | None = Field(
        default=None,
        description="Explicit filename when provided; None means the caller should generate one.",
    )


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
