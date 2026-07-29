from __future__ import annotations

import unittest

from pydantic import ValidationError

from governance.schemas.taxonomy import Taxonomy


class TaxonomyTest(unittest.TestCase):
    """
    Pins taxonomy validation.
    """

    def test_empty_domain_is_rejected(self) -> None:
        """
        A taxonomy must classify at least one domain prefix.
        """

        with self.assertRaises(ValidationError):
            Taxonomy(domain=())

    def test_duplicate_domain_prefix_is_rejected(self) -> None:
        """
        Duplicate domain prefixes make the taxonomy ambiguous and are rejected.
        """

        with self.assertRaises(ValidationError):
            Taxonomy(domain=("fathom.schemas", "fathom.schemas"))

    def test_valid_taxonomy_defaults_to_provisional(self) -> None:
        """
        A taxonomy is provisional until D1 marks it otherwise.
        """

        taxonomy = Taxonomy(domain=("fathom.schemas",))

        self.assertTrue(taxonomy.provisional)
