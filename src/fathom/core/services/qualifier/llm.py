from __future__ import annotations

import json
from typing import Optional

from pydantic import ValidationError

from fathom.constants.qualification import (
    DEFAULT_REJECTION_MESSAGE,
    QualificationLabel,
    RationaleCategory,
)
from fathom.core.prompts.qualifier import (
    GeminiQualifierPromptBuilder,
    QualifierPromptBuilder,
)
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.schemas.configuration import QualifierConfiguration
from fathom.schemas.qualification import QualificationVerdict, Rationale
from fathom.utils.parsing import strip_code_fences


class LLMIntentQualifier(IntentQualifierPort):
    """
    Qualifier that delegates the executability judgement to an LLM.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        message: str = DEFAULT_REJECTION_MESSAGE,
        configuration: Optional[QualifierConfiguration] = None,
        prompt_builder: Optional[QualifierPromptBuilder] = None,
    ) -> None:
        """
        Wire the qualifier with an LLM port. Lifecycle of `llm` is the caller's.
        """

        self.__llm = llm
        self.__rejection_message = message
        self.__configuration = configuration or QualifierConfiguration()
        self.__prompt_builder = prompt_builder or GeminiQualifierPromptBuilder()

    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Classify the intent's executability and return a verdict.

        Fail-open semantics: any LLM, parse, or schema failure yields an EXECUTABLE verdict tagged with QUALIFIER_ERROR
        so the gate explicitly passes the run through; the qualifier can never cause a false rejection.
        """

        normalized = intent.strip() if intent else ""

        if not normalized:
            return self.__empty_intent_verdict()

        try:
            response = await self.__llm.generate(
                use_cache=self.__configuration.inference.use_cache,
                prompt=[self.__prompt_builder.build_user_prompt(intent=normalized)],
                system_instruction=self.__prompt_builder.build_system_instruction(),
            )
        except Exception as exception:
            return self.__fail_open(reason=f"llm_error: {exception}")

        return self.__parse_verdict(content=response.content)

    def __parse_verdict(self, *, content: str) -> QualificationVerdict:
        """
        Parse the LLM response into a QualificationVerdict; fail open on any issue.
        """

        try:
            payload = json.loads(strip_code_fences(content))
        except json.JSONDecodeError as exception:
            return self.__fail_open(reason=f"non_json_response: {exception}")

        if not isinstance(payload, dict):
            return self.__fail_open(
                reason=f"non_object_json_response: payload_type={type(payload).__name__}"
            )

        try:
            rationale_payload = payload.get("rationale") or {}

            if not isinstance(rationale_payload, dict):
                raise TypeError("rationale field is not an object")

            return QualificationVerdict(
                message=None,
                confidence=float(payload["confidence"]),
                label=QualificationLabel(payload["label"]),
                rationale=Rationale(
                    reasoning=str(rationale_payload.get("reasoning", "")),
                    category=RationaleCategory(rationale_payload.get("category", "unspecified")),
                ),
            )
        except (KeyError, ValidationError, ValueError, TypeError) as exception:
            return self.__fail_open(reason=f"schema_validation_failed: {exception}")

    def __empty_intent_verdict(self) -> QualificationVerdict:
        """
        Deterministic verdict for an empty / whitespace-only intent.
        """

        return QualificationVerdict(
            confidence=1.0,
            message=self.__rejection_message,
            label=QualificationLabel.NOT_EXECUTABLE,
            rationale=Rationale(
                category=RationaleCategory.EMPTY,
                reasoning="Empty or whitespace-only intent cannot be executed.",
            ),
        )

    def __fail_open(self, *, reason: str) -> QualificationVerdict:
        """
        Return a non-blocking verdict recording why the qualifier abstained.
        """

        return QualificationVerdict(
            message=None,
            confidence=0.5,
            label=QualificationLabel.EXECUTABLE,
            rationale=Rationale(
                category=RationaleCategory.QUALIFIER_ERROR,
                reasoning=f"Qualifier abstained ({reason}); allowing execution to proceed.",
            ),
        )
