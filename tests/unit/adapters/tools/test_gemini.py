from __future__ import annotations

import unittest

import fathom.adapters.tools as tools
from fathom.adapters.tools import GeminiToolDialect


class GeminiToolDialectTest(unittest.TestCase):
    """
    Gemini tool translation remains self-contained in the provider-neutral foundation.
    """

    def test_package_exports_the_gemini_dialect(self) -> None:
        """
        The adapter package continues to expose the Gemini dialect after adding providers.
        """

        self.assertIn("GeminiToolDialect", tools.__all__)
        self.assertIs(tools.GeminiToolDialect, GeminiToolDialect)

    def test_encode_wraps_canonical_declarations_without_mutating_them(self) -> None:
        """
        Gemini translation adds its SDK envelope and leaves the canonical declaration intact.
        """

        declaration = {"name": "execute_ui", "parameters": {"type": "object"}}

        encoded = GeminiToolDialect().encode(payload={"tools": [declaration]})

        self.assertEqual(encoded, {"function_declarations": [declaration]})
        self.assertEqual(declaration, {"name": "execute_ui", "parameters": {"type": "object"}})
