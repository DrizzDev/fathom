from __future__ import annotations

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
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.qualification import QualificationVerdict, Rationale


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
        self.__structured_output = StructuredOutput(payload=QualificationVerdict)

    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Classify the intent's executability and return a verdict.

        Fail-open semantics: any LLM, parse, or schema failure yields an EXECUTABLE verdict tagged
        with QUALIFIER_ERROR so the gate explicitly passes the run through; the qualifier can never
        cause a false rejection.
        """

        normalized = intent.strip() if intent else ""

        if not normalized:
            return self.__empty_intent_verdict()

        try:
            response = await self.__llm.generate(
                use_cache=self.__configuration.inference.use_cache,
                prompt=[self.__prompt_builder.build_user_prompt(intent=normalized)],
                system_instruction=self.__prompt_builder.build_system_instruction(),
                structured_output=self.__structured_output,
            )
        except Exception as exception:
            return self.__fail_open(reason=f"llm_error: {exception}")

        return self.__parse_verdict(content=response.content)

    def __parse_verdict(self, *, content: str) -> QualificationVerdict:
        """
        Validate the structured-output payload; fail open on any defect.
        """

        try:
            verdict = QualificationVerdict.model_validate_json(content)
        except ValidationError as exception:
            return self.__fail_open(reason=f"schema_validation_failed: {exception}")
        except (ValueError, TypeError) as exception:
            return self.__fail_open(reason=f"non_json_response: {exception}")

        # The qualifier owns the user-facing rejection message; the LLM never sets it.
        return verdict.model_copy(update={"message": None})

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
