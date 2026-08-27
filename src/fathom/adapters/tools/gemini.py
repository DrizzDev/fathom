"""Encode provider-neutral tools for the Gemini SDK."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from fathom.interfaces.tools import ToolDeclaration, ToolDialect


class GeminiToolDialect(ToolDialect):
    """
    Encode the provider-neutral tool contract as Gemini function declarations.
    """

    def encode(self, *, payload: Mapping[str, List[ToolDeclaration]]) -> Dict[str, Any]:
        """
        Return the Gemini SDK envelope without mutating the canonical declarations.
        """

        return {"function_declarations": list(payload.get("tools", []))}
