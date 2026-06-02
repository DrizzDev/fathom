from __future__ import annotations

import json
import time
from logging import getLogger
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

logger = getLogger(__name__)


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
        Wire the qualifier with an LLM port, gate configuration, and an optional prompt builder.
        """

        self.__llm = llm
        self.__rejection_message = message
        self.__configuration = configuration or QualifierConfiguration()
        self.__prompt_builder = prompt_builder or GeminiQualifierPromptBuilder()

    async def qualify(self, *, intent: str) -> QualificationVerdict:
        """
        Classify the intent's executability and return a verdict.

        Fail-open semantics: any LLM, parse, or schema failure yields a PROBABLY_EXECUTABLE
        verdict so the qualifier can never cause a false rejection.
        """

        normalized = intent.strip() if intent else ""
        logger.info("qualifier.qualify_start", extra={"intent_length": len(normalized)})

        if not normalized:
            logger.warning("qualifier.empty_intent")
            return self.__empty_intent_verdict()

        started_at = time.perf_counter()

        try:
            response = await self.__llm.generate(
                use_cache=self.__configuration.use_cache,
                prompt=[self.__prompt_builder.build_user_prompt(intent=normalized)],
                system_instruction=self.__prompt_builder.build_system_instruction(),
            )
        except Exception as exception:
            latency = time.perf_counter() - started_at
            logger.warning(
                "qualifier.llm_call_failed",
                extra={"latency": latency, "reason": str(exception)},
            )
            return self.__fail_open(reason=f"llm_error: {exception}")

        latency = time.perf_counter() - started_at
        logger.info(
            "qualifier.llm_response",
            extra={"latency": latency, "response_size": len(response.content or "")},
        )

        verdict = self.__parse_verdict(content=response.content)

        logger.info(
            "qualifier.qualify_done",
            extra={
                "label": verdict.label.value,
                "confidence": verdict.confidence,
                "category": verdict.rationale.category.value,
                "blocked": verdict.should_block(floor=self.__configuration.confidence_floor),
            },
        )

        return verdict

    def __parse_verdict(self, *, content: str) -> QualificationVerdict:
        """
        Parse the LLM response into a QualificationVerdict; fail open on any issue.
        """

        try:
            payload = json.loads(strip_code_fences(content))
        except json.JSONDecodeError as exception:
            logger.warning("qualifier.non_json_response", extra={"reason": str(exception)})
            return self.__fail_open(reason="non_json_response")

        try:
            rationale_payload = payload.get("rationale") or {}
            verdict = QualificationVerdict(
                message=None,
                confidence=float(payload["confidence"]),
                label=QualificationLabel(payload["label"]),
                rationale=Rationale(
                    reasoning=str(rationale_payload.get("reasoning", "")),
                    category=RationaleCategory(rationale_payload.get("category", "unspecified")),
                ),
            )
        except (KeyError, ValidationError, ValueError, TypeError) as exception:
            logger.warning("qualifier.schema_validation_failed", extra={"reason": str(exception)})
            return self.__fail_open(reason="schema_validation_failed")

        if verdict.should_block(floor=self.__configuration.confidence_floor):
            return verdict.model_copy(update={"message": self.__rejection_message})

        return verdict

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
            label=QualificationLabel.PROBABLY_EXECUTABLE,
            rationale=Rationale(
                category=RationaleCategory.QUALIFIER_ERROR,
                reasoning=f"Qualifier abstained ({reason}); allowing execution to proceed.",
            ),
        )
