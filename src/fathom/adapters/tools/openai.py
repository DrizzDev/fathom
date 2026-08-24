from __future__ import annotations

from typing import Any, Dict, List, Mapping

from fathom.interfaces.tools import ToolDeclaration, ToolDialect


class OpenAIToolDialect(ToolDialect):
    """
    Encode Fathom's canonical tool declarations for OpenAI Chat Completions.
    """

    def encode(self, *, payload: Mapping[str, List[ToolDeclaration]]) -> Dict[str, Any]:
        """
        Wrap each canonical function in OpenAI's function-tool envelope.
        """

        return {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": declaration["name"],
                        "description": declaration.get("description", ""),
                        "parameters": self.__schema(
                            schema=declaration.get(
                                "parameters", {"type": "object", "properties": {}}
                            )
                        ),
                    },
                }
                for declaration in payload.get("tools", [])
            ]
        }

    @classmethod
    def __schema(cls, *, schema: Any) -> Any:
        """
        Normalize canonical/Gemini JSON Schema types for OpenAI tools.
        """

        if isinstance(schema, list):
            return [cls.__schema(schema=item) for item in schema]
        if not isinstance(schema, dict):
            return schema

        normalized = {key: cls.__schema(schema=value) for key, value in schema.items()}
        if isinstance(normalized.get("type"), str):
            normalized["type"] = normalized["type"].lower()
        return normalized
