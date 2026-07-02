from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import Field, JsonValue

from fathom.schemas.conversation.base import ConversationSchema


class FunctionCallPart(ConversationSchema):
    """
    A tool/function call within a conversation turn.
    """

    name: str = Field(description="Tool function name")
    args: Dict[str, JsonValue] = Field(default_factory=dict, description="Function arguments")


class TurnPart(ConversationSchema):
    """
    A single content part within a conversation turn.

    Exactly one of text, image_data, or function_call should be set.
    """

    text: Optional[str] = Field(default=None, description="Plain text content for the part.")
    image_data: Optional[bytes] = Field(
        default=None,
        description="Raw image bytes when the part carries an inline image.",
    )
    mime_type: Optional[str] = Field(
        default=None,
        description="Media type that describes `image_data` when set.",
    )
    function_call: Optional[FunctionCallPart] = Field(
        default=None,
        description="Tool/function invocation when the part is a tool call.",
    )

    @classmethod
    def from_text(cls, text: str) -> TurnPart:
        """
        Build a turn part carrying only plain text content.
        """

        return cls(text=text)

    @classmethod
    def from_image(cls, data: bytes, mime_type: str = "image/png") -> TurnPart:
        """
        Build a turn part carrying inline image bytes with an explicit media type.
        """

        return cls(image_data=data, mime_type=mime_type)

    @classmethod
    def from_function_call(cls, name: str, args: Dict[str, JsonValue]) -> TurnPart:
        """
        Build a turn part wrapping a tool-call request with its bound arguments.
        """

        return cls(function_call=FunctionCallPart(name=name, args=args))


class ConversationTurn(ConversationSchema):
    """
    A single turn in a multi-turn conversation.

    Provider-neutral representation that adapters convert to/from their native types (e.g. google.genai.types.Content).
    """

    role: Literal["user", "model"] = Field(description="Speaker role")
    parts: List[TurnPart] = Field(default_factory=list, description="Content parts")
