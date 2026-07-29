from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from governance.errors import GovernanceError
from governance.schemas.manifest import Manifest
from governance.schemas.taxonomy import Taxonomy


class ConfigLoader:
    """
    Loads the governance configuration files, failing closed when either is absent.
    """

    def manifest(self, *, path: Path) -> Manifest:
        """
        Load the debt manifest from a JSON file.
        """

        return Manifest.model_validate(self.__read(path=path))

    def taxonomy(self, *, path: Path) -> Taxonomy:
        """
        Load the package taxonomy from a JSON file.
        """

        return Taxonomy.model_validate(self.__read(path=path))

    @staticmethod
    def __read(*, path: Path) -> Any:
        """
        Parse a JSON configuration file, raising a governance error when it is missing.
        """

        if not path.exists():
            raise GovernanceError(f"governance configuration missing at {path}")

        return json.loads(path.read_text())
