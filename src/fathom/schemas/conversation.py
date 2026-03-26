"""
Provider-neutral conversation models for multi-turn LLM interactions.

These models decouple core domain logic from provider-specific types
(e.g. google.genai.types.Content), ensuring the LLMPort interface
boundary remains clean.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class FunctionCallPart(BaseModel):
    """A tool/function call within a conversation turn."""

    name: str = Field(description="Tool function name")
    args: Dict[str, Any] = Field(default_factory=dict, description="Function arguments")


class TurnPart(BaseModel):
    """
    A single content part within a conversation turn.

    Exactly one of text, image_data, or function_call should be set.
    """

    text: Optional[str] = None
    image_data: Optional[bytes] = None
    mime_type: Optional[str] = None
    function_call: Optional[FunctionCallPart] = None

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_text(cls, text: str) -> TurnPart:
        return cls(text=text)

    @classmethod
    def from_image(cls, data: bytes, mime_type: str = "image/png") -> TurnPart:
        return cls(image_data=data, mime_type=mime_type)

    @classmethod
    def from_function_call(cls, name: str, args: Dict[str, Any]) -> TurnPart:
        return cls(function_call=FunctionCallPart(name=name, args=args))


class ConversationTurn(BaseModel):
    """
    A single turn in a multi-turn conversation.

    Provider-neutral representation that adapters convert to/from
    their native types (e.g. google.genai.types.Content).
    """

    role: Literal["user", "model"] = Field(description="Speaker role")
    parts: List[TurnPart] = Field(default_factory=list, description="Content parts")
