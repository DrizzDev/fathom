from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from governance.configuration import ConfigLoader
from governance.constants import GovernanceMode
from governance.errors import GovernanceError
from governance.schemas.manifest import Manifest
from governance.schemas.taxonomy import Taxonomy


class ConfigLoaderTest(unittest.TestCase):
    """
    Pins fail-closed loading of the governance configuration files.
    """

    def test_missing_manifest_fails_closed(self) -> None:
        """
        Loading an absent manifest raises rather than defaulting to a permissive report mode.
        """

        with self.assertRaises(GovernanceError):
            ConfigLoader().manifest(path=Path("/nonexistent/governance/debt.json"))

    def test_missing_taxonomy_fails_closed(self) -> None:
        """
        Loading an absent taxonomy raises rather than scanning nothing.
        """

        with self.assertRaises(GovernanceError):
            ConfigLoader().taxonomy(path=Path("/nonexistent/governance/taxonomy.json"))

    def test_loads_manifest_and_taxonomy(self) -> None:
        """
        Well-formed configuration files load into validated models.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "debt.json"
            taxonomy_path = root / "taxonomy.json"
            manifest_path.write_text(Manifest(mode=GovernanceMode.RATCHET).model_dump_json())
            taxonomy_path.write_text(
                Taxonomy(provisional=False, domain=("fathom.schemas",)).model_dump_json()
            )

            loader = ConfigLoader()

            self.assertIs(loader.manifest(path=manifest_path).mode, GovernanceMode.RATCHET)
            self.assertFalse(loader.taxonomy(path=taxonomy_path).provisional)
