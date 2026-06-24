from __future__ import annotations

import re
from datetime import datetime
from logging import getLogger
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fathom.schemas.artifact import ArtifactKind
    from fathom.settings.env import FathomSettings

logger = getLogger(__name__)


class SharedPathManager:
    """
    Manages base paths for assets, memory, and logs.
    Enforces structure: assets/{category}/{date}/{session}/
    """

    def __init__(self, settings: "FathomSettings") -> None:
        """
        Initialize path manager with settings.
        """

        self.__base_path = settings.assets_path
        self.__base_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Fathom asset paths resolved",
            extra={
                "component": "base.paths",
                "event": "fathom.paths.resolved",
                "base_path": str(self.__base_path),
                "memory_path": str(self.__base_path / "memory"),
            },
        )

    def __get_category_root(self, category: str, session_id: str) -> Path:
        """
        Get the root directory for a specific category within a session.
        """

        date_str = datetime.now().strftime("%Y-%m-%d")
        path = self.__base_path / category / date_str / session_id
        path.mkdir(parents=True, exist_ok=True)

        return path

    def get_screenshot_path(self, *, session_id: str, filename: str) -> Path:
        """
        Get the full path for a screenshot.
        """

        directory = self.__get_category_root("screenshot", session_id)
        return directory / filename

    def get_trace_path(self, *, session_id: str, filename: str) -> Path:
        """
        Get the full path for a trace image.
        """

        directory = self.__get_category_root("traces", session_id)
        return directory / filename

    def get_history_directory(self, *, session_id: str) -> Path:
        """
        Get directory for history artifacts.
        """

        return self.__get_category_root("history", session_id)

    def get_report_directory(self, *, session_id: str) -> Path:
        """
        Get directory for exploration graph exports and reports.
        """

        return self.__get_category_root("reports", session_id)

    def get_annotated_path(self, *, session_id: str, filename: str) -> Path:
        """
        Get the full path for an annotated image.
        """

        directory = self.__get_category_root("annotated", session_id)
        return directory / filename

    def get_xml_path(self, *, session_id: str, filename: str) -> Path:
        """
        Get the full path for an XML dump.
        """

        directory = self.__get_category_root("xmls", session_id)
        return directory / filename

    def get_artifact_path(
        self,
        *,
        kind: "ArtifactKind",
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

    def get_knowledge_db_path(self, *, package: Optional[str] = None) -> Path:
        """
        Path to the knowledge graph database, namespaced per package when given.

        A per-package database keeps each app's screens, transitions, and defects
        isolated, so a report never mixes apps and a run never resumes into another
        app's frontier. Without a package the shared database path is returned.
        """

        if not package:
            return self.memory_path / "knowledge.db"

        directory = self.memory_path / "knowledge"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self.__safe_package(package)}.db"

    @staticmethod
    def __safe_package(package: str) -> str:
        """
        Reduce a package identifier to a filename-safe stem.
        """

        return re.sub(r"[^A-Za-z0-9._-]", "_", package)

    def get_ledger_db_path(self) -> Path:
        """
        Path to ledger database.
        """

        return self.memory_path / "ledger.db"

    def get_checkpoint_directory(self) -> Path:
        """
        Directory containing all per-workflow LangGraph checkpoint database files.
        """

        path = self.memory_path / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)

        return path

    def get_checkpoint_path(self, *, workflow_id: Optional[str]) -> Path:
        """
        Per-workflow LangGraph checkpoint database path under the dedicated checkpoint directory.
        """

        identifier = workflow_id.strip() or "default" if workflow_id else "default"

        return self.get_checkpoint_directory() / f"checkpoints__{identifier}.db"

    def get_exploration_checkpoint_path(self, *, workflow_id: Optional[str]) -> Path:
        """
        Per-workflow exploration DFS checkpoint database path, kept separate from
        the LangGraph checkpoint file so the two never collide.
        """

        identifier = workflow_id.strip() or "default" if workflow_id else "default"

        return self.get_checkpoint_directory() / f"exploration__{identifier}.db"
