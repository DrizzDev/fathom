"""Define provider-neutral tool declaration contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping

ToolDeclaration = Dict[str, Any]
CanonicalToolPayload = Dict[str, List[ToolDeclaration]]


class ToolDialect(ABC):
    """
    Translate Fathom's canonical tool contract into one provider's wire format.

    The canonical payload deliberately contains only the tool declarations. Provider-specific
    envelope keys such as Gemini's ``function_declarations`` do not cross the LLM port boundary.
    """

    @abstractmethod
    def encode(self, *, payload: Mapping[str, List[ToolDeclaration]]) -> Dict[str, Any]:
        """
        Encode canonical declarations for the provider SDK.
        """

        raise NotImplementedError
