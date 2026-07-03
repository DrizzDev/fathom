from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Final, List, Optional, Tuple

from fathom.base.paths import SharedPathManager
from fathom.constants.collaboration import ArtifactKind
from fathom.constants.conversation import SCRIPT_CONTENT_FILENAME

if TYPE_CHECKING:
    from os import stat_result
    from pathlib import Path


class ArtifactCatalog:
    """
    Stateless artifact discovery and classification over the shared path manager.

    Both the workflow runner and the graph nodes use this to find generated files for one workflow, and to classify each one for recording into the interaction layer.
    Discovery is sorted by capture time (file mtime) so the timeline reflects the order in which the host adapter wrote the files.
    """

    __STEP_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:^|_)step_([0-9]+)(?:_|\.)")

    __CATEGORY_FOLDERS: Final[Tuple[str, ...]] = (
        "xmls",
        "history",
        "traces",
        "screenshot",
        "annotated",
    )

    def __init__(self, *, path_manager: SharedPathManager) -> None:
        """
        Bind the catalog to one shared path manager.
        """

        self.__path_manager = path_manager

    async def discover(
        self,
        *,
        workflow: str,
        package_name: str,
        only_step: Optional[int] = None,
        only_workflow_scope: bool = False,
    ) -> List[Tuple[Path, stat_result]]:
        """
        Return existing artifact paths sorted by capture time.

        only_step           -> only files whose filename encodes this step.
        only_workflow_scope -> only files whose filename has no step prefix.
        """

        records: List[Tuple[Path, stat_result]] = []

        for path in self.__discover_paths(package_name=package_name, workflow=workflow):
            step_number = self.step_number(path=path)
            if only_step is not None and step_number != only_step:
                continue

            if only_workflow_scope and step_number is not None:
                continue

            try:
                stat = await asyncio.to_thread(path.stat)
            except FileNotFoundError:
                continue

            records.append((path, stat))

        return sorted(records, key=lambda item: (item[1].st_mtime, str(item[0])))

    def step_number(self, *, path: Path) -> Optional[int]:
        """
        Extract a step number from standard artifact filenames.
        """

        match = self.__STEP_PATTERN.search(path.name)
        return int(match.group(1)) if match else None

    def category(self, *, path: Path) -> str:
        """
        Return the artifact category folder name.
        """

        return path.parents[3].name if len(path.parents) > 3 else ""

    def package(self, *, path: Path) -> Optional[str]:
        """
        Return the package folder embedded in the artifact path.
        """

        if len(path.parents) <= 2:
            return None

        return path.parents[1].name

    def kind(self, *, path: Path) -> Optional[ArtifactKind]:
        """
        Resolve interaction artifact kind, or None when the path is not a public artifact.
        """

        if self.is_script(path=path):
            return None

        category = self.category(path=path)

        if category in {"traces"}:
            return ArtifactKind.TRACE

        if category in {"screenshot", "annotated"}:
            return ArtifactKind.SCREENSHOT

        return ArtifactKind.STRUCTURED_LOG

    def is_script(self, *, path: Path) -> bool:
        """
        Return whether the path contains generated script content.
        """

        return path.name == SCRIPT_CONTENT_FILENAME or "__script__" in path.stem

    def mime(self, *, path: Path) -> str:
        """
        Resolve media type from a generated artifact path.
        """

        suffix = path.suffix.lower()

        if suffix == ".png":
            return "image/png"

        if suffix == ".xml":
            return "application/xml"

        if suffix in {".yaml", ".yml"}:
            return "application/yaml"

        if suffix == ".txt":
            return "text/plain"

        return "application/json"

    def retention(self, *, path: Path) -> Optional[str]:
        """
        Resolve retention class for a generated artifact.
        """

        _ = path

        return None

    def __discover_paths(self, *, package_name: str, workflow: str) -> List[Path]:
        """
        Find generated artifacts for one workflow across all observed packages.
        """

        paths: List[Path] = []

        for category in self.__CATEGORY_FOLDERS:
            root = self.__path_manager.base_path / category
            if not root.exists():
                continue

            paths.extend(path for path in root.glob(f"*/*/{workflow}/*") if path.is_file())

        return sorted(
            paths,
            key=lambda path: (
                0 if self.package(path=path) == package_name else 1,
                str(path),
            ),
        )
