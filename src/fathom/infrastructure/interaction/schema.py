from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AuditColumns:
    """
    Renders shared SQL audit columns for interaction storage tables.
    """

    timestamp_type: str
    metadata_type: str
    indent: str

    def created(self) -> str:
        """
        Render created and metadata audit columns.
        """

        return self.__render(updated=False, deleted=False)

    def created_deleted(self) -> str:
        """
        Render created, deleted, and metadata audit columns.
        """

        return self.__render(updated=False, deleted=True)

    def created_updated(self) -> str:
        """
        Render created, updated, and metadata audit columns.
        """

        return self.__render(updated=True, deleted=False)

    def created_updated_deleted(self) -> str:
        """
        Render created, updated, deleted, and metadata audit columns.
        """

        return self.__render(updated=True, deleted=True)

    def __render(self, *, updated: bool, deleted: bool) -> str:
        """
        Build the ordered audit column block for one table shape.
        """

        columns: List[str] = [f"created_at {self.timestamp_type} NOT NULL"]
        if updated:
            columns.append(f"updated_at {self.timestamp_type} NOT NULL")
        if deleted:
            columns.append(f"deleted_at {self.timestamp_type}")
        columns.append(f"metadata {self.metadata_type} NOT NULL")

        return "\n".join(f"{self.indent}{column}," for column in columns[:-1]) + (
            f"\n{self.indent}{columns[-1]}"
        )
