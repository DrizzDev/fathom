from __future__ import annotations

from datetime import datetime
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from fathom.settings.env import FathomSettings

if TYPE_CHECKING:
    from fathom.schemas.artifact import ArtifactKind


class SharedPathManager:
    """
    Manages base paths for assets, memory, and logs.
    Enforces structure: assets/{category}/{date}/{package}/{session}/
    """

    def __init__(self, settings: FathomSettings) -> None:
        """
        Initialize path manager with settings.
        """

        self.__base_path = settings.assets_path
        self.__base_path.mkdir(parents=True, exist_ok=True)

    def __get_category_root(self, category: str, package_name: str, session_id: str) -> Path:
        """
        Get the root directory for a specific category within a session.
        """

        date_str = datetime.now().strftime("%Y-%m-%d")
        path = self.__base_path / category / date_str / package_name / session_id
        path.mkdir(parents=True, exist_ok=True)

        return path

    def get_screenshot_path(self, package_name: str, session_id: str, filename: str) -> Path:
        """
        Get the full path for a screenshot.
        """

        directory = self.__get_category_root("screenshot", package_name, session_id)
        return directory / filename

    def get_trace_path(self, package_name: str, session_id: str, filename: str) -> Path:
        """
        Get the full path for a trace image.
        """

        directory = self.__get_category_root("traces", package_name, session_id)
        return directory / filename

    def get_history_directory(self, package_name: str, session_id: str) -> Path:
        """
        Get directory for history artifacts.
        """

        return self.__get_category_root("history", package_name, session_id)

    def get_annotated_path(self, package_name: str, session_id: str, filename: str) -> Path:
        """
        Get the full path for an annotated image.
        """

        directory = self.__get_category_root("annotated", package_name, session_id)
        return directory / filename

    def get_xml_path(self, package_name: str, session_id: str, filename: str) -> Path:
        """
        Get the full path for an XML dump.
        """

        directory = self.__get_category_root("xmls", package_name, session_id)
        return directory / filename

    def get_artifact_path(
        self,
        *,
        kind: "ArtifactKind",
        package_name: str,
        session_id: str,
        filename: str,
    ) -> Path:
        """
        Resolve the EFS path for one artifact, keyed by :class:`ArtifactKind`.

        New artifact pipeline callers route through this method so the
        layout stays a single source of truth instead of being scattered
        across kind-specific helpers.
        """

        directory = self.__get_category_root(
            self.__directory_for(kind=kind),
            package_name,
            session_id,
        )
        return directory / filename

    @staticmethod
    def __directory_for(*, kind: "ArtifactKind") -> str:
        """
        Resolve the canonical directory for one :class:`ArtifactKind`
        through the shared :class:`ArtifactCategory` mapping so the
        EFS layout cannot drift from the cloud-side category.
        """

        from fathom.schemas.artifact import ArtifactCategory

        return ArtifactCategory.for_(kind=kind)

    @property
    def base_path(self) -> Path:
        """
        Root directory for all assets.
        """

        return self.__base_path

    @property
    def memory_path(self) -> Path:
        """
        Directory for persistent memory (SQLite DBs).
        """

        path = self.__base_path / "memory"
        path.mkdir(parents=True, exist_ok=True)

        return path

    def get_knowledge_db_path(self) -> Path:
        """
        Path to knowledge graph database.
        """

        return self.memory_path / "knowledge.db"

    def get_ledger_db_path(self) -> Path:
        """
        Path to ledger database.
        """

        return self.memory_path / "ledger.db"
