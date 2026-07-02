from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import Field

from fathom.schemas.conversation.base import ConversationSchema


class WireObservation(ConversationSchema):
    """
    Chat-facing observation projection carrying the screen summary and evidence.
    """

    summary: Optional[str] = Field(default=None, description="Screen observation summary.")
    evidence: Optional[str] = Field(default=None, description="Screen evidence for the step.")


class WireProgressBody(ConversationSchema):
    """
    Chat-facing compact body for per-step progress messages.
    """

    summary: str = Field(description="Chat bubble summary for this step.")
    rationale: Optional[str] = Field(default=None, description="Reason the action was selected.")
    observation: Optional[WireObservation] = Field(
        default=None,
        description="Screen observation projection, when the stored body carries one.",
    )

    @classmethod
    def project(cls, *, body: Mapping[str, Any]) -> Optional["WireProgressBody"]:
        """
        Project a stored progress body into the compact chat shape.
        """

        summary = body.get("summary")
        if not isinstance(summary, str) or not summary:
            return None

        return cls(
            summary=summary,
            rationale=cls.__string(raw=body.get("rationale")),
            observation=cls.__observation(raw=body.get("observation")),
        )

    @staticmethod
    def __string(*, raw: Any) -> Optional[str]:
        """
        Return raw only when it is a non-empty string.
        """

        return raw if isinstance(raw, str) and raw else None

    @classmethod
    def __observation(cls, *, raw: Any) -> Optional[WireObservation]:
        """
        Pass through the observation summary and evidence fields verbatim when the source is a dict.
        """

        if not isinstance(raw, dict):
            return None

        return WireObservation(
            summary=cls.__string(raw=raw.get("summary")),
            evidence=cls.__string(raw=raw.get("evidence")),
        )


class WireRequestBody(ConversationSchema):
    """
    Chat-facing compact body for user request messages that start an intent.
    """

    intent: str = Field(description="User-supplied intent text.")

    @classmethod
    def project(cls, *, body: Mapping[str, Any]) -> Optional["WireRequestBody"]:
        """
        Project a stored request body into the compact chat shape.
        """

        intent = body.get("intent")
        if not isinstance(intent, str) or not intent:
            return None

        return cls(intent=intent)


class WireResultBody(ConversationSchema):
    """
    Chat-facing compact body for terminal result messages.
    """

    success: bool = Field(description="Whether the run reached its goal.")
    summary: str = Field(description="Chat bubble summary for the run outcome.")
    reason: Optional[str] = Field(default=None, description="Terminal outcome reason.")

    @classmethod
    def project(cls, *, body: Mapping[str, Any]) -> Optional["WireResultBody"]:
        """
        Project a stored result body into the compact chat shape.
        """

        summary = body.get("summary")
        success = body.get("success")

        if not isinstance(summary, str) or not isinstance(success, bool):
            return None

        reason = body.get("reason")

        return cls(
            summary=summary,
            success=success,
            reason=reason if isinstance(reason, str) else None,
        )


class WireQuestionBody(ConversationSchema):
    """
    Chat-facing compact body for HITL prompt messages.
    """

    question: str = Field(description="Prompt text shown to the user.")

    @classmethod
    def project(cls, *, body: Mapping[str, Any]) -> Optional["WireQuestionBody"]:
        """
        Project a stored question body into the compact chat shape.
        """

        question = body.get("question")
        if not isinstance(question, str) or not question:
            return None

        return cls(question=question)


class WireAnswerBody(ConversationSchema):
    """
    Chat-facing compact body for HITL response messages.
    """

    answer: str = Field(description="User-supplied answer text.")

    @classmethod
    def project(cls, *, body: Mapping[str, Any]) -> Optional["WireAnswerBody"]:
        """
        Project a stored answer body into the compact chat shape.
        """

        answer = body.get("answer")
        if not isinstance(answer, str) or not answer:
            return None

        return cls(answer=answer)
