from __future__ import annotations

import unittest

from fathom.adapters.tools import OpenAIToolDialect


class OpenAIToolDialectTest(unittest.TestCase):
    """
    OpenAI tool translation stays at the provider adapter boundary.
    """

    def test_schema_types_are_normalized_for_openai(self) -> None:
        """
        Gemini-style uppercase schema types become OpenAI lowercase types.
        """

        encoded = OpenAIToolDialect().encode(
            payload={
                "tools": [
                    {
                        "name": "execute_ui",
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {"x": {"type": "NUMBER"}},
                        },
                    }
                ]
            }
        )

        parameters = encoded["tools"][0]["function"]["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(parameters["properties"]["x"]["type"], "number")
