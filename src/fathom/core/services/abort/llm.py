from __future__ import annotations

import json
from logging import getLogger
from typing import Optional

from pydantic import ValidationError

from fathom.constants.abort import ABORT_WARMUP_PROMPT
from fathom.core.prompts.abort import AbortPromptBuilder, GeminiAbortPromptBuilder
from fathom.interfaces.abort import AbortDetectorPort
from fathom.interfaces.llm import LLMPort
from fathom.schemas.abort import (
    AbortDecision,
    AbortDetectorConfiguration,
    AbortDetectorResponse,
)
from fathom.schemas.llm import StructuredOutput

logger = getLogger(__name__)


class LLMAbortDetector(AbortDetectorPort):
    """
    Abort detector implementation backed by an LLM port.
    """

    def __init__(
        self,
        *,
        llm: LLMPort,
        prompt_builder: Optional[AbortPromptBuilder] = None,
        configuration: Optional[AbortDetectorConfiguration] = None,
    ) -> None:
        """
        Wire the detector with an LLM port and optional configuration / prompt builder.
        """

        self.__llm = llm
        self.__configuration = configuration or AbortDetectorConfiguration()
        self.__prompt_builder = prompt_builder or GeminiAbortPromptBuilder()
        self.__structured_output = StructuredOutput(payload=AbortDetectorResponse)

    async def aborted(self, *, response: str) -> AbortDecision:
        """
        Classify the response via the LLM and fail-open on any error or schema violation.
        """

        normalized = response.strip()

        if not normalized:
            return self.__fail_open()

        try:
            llm_response = await self.__llm.generate(
                use_cache=self.__configuration.inference.use_cache,
                system_instruction=self.__prompt_builder.build_system_instruction(),
                prompt=[self.__prompt_builder.build_user_prompt(response=normalized)],
                structured_output=self.__structured_output,
            )
        except Exception as exception:
            logger.warning(
                "Abort detector LLM call failed; falling back",
                extra={
                    "event": "abort.detector.llm.failed",
                    "component": "core.services.abort.llm",
                    "error.message": str(exception),
                    "error.kind": type(exception).__name__,
                },
            )
            return self.__fail_open()

        return self.__parse_decision(content=llm_response.content)

    async def warmup(self) -> None:
        """
        Issue a tiny request to prime the LLM model so the first real call is fast.
        """

        try:
            await self.__llm.generate(
                use_cache=False,
                prompt=[ABORT_WARMUP_PROMPT],
                system_instruction=self.__prompt_builder.build_system_instruction(),
                structured_output=self.__structured_output,
            )
        except Exception as exception:
            logger.info(
                "Abort detector warmup failed; first real call may be slower",
                extra={
                    "event": "abort.detector.warmup.failed",
                    "component": "core.services.abort.llm",
                    "error.kind": type(exception).__name__,
                },
            )

    def __parse_decision(self, *, content: str) -> AbortDecision:
        """
        Validate the structured-output payload and fail-open on any defect.
        """

        try:
            parsed = AbortDetectorResponse.model_validate_json(content)
        except ValidationError as exception:
            logger.warning(
                "Abort detector returned un-parseable response; falling back",
                extra={
                    "event": "abort.detector.parse_failed",
                    "component": "core.services.abort.llm",
                    "error.kind": type(exception).__name__,
                    "content.preview": (content or "")[:120],
                },
            )
            return self.__fail_open()
        except (json.JSONDecodeError, ValueError, TypeError) as exception:
            logger.warning(
                "Abort detector returned un-parseable response; falling back",
                extra={
                    "event": "abort.detector.parse_failed",
                    "component": "core.services.abort.llm",
                    "error.kind": type(exception).__name__,
                    "content.preview": (content or "")[:120],
                },
            )
            return self.__fail_open()

        floor = self.__configuration.confidence.floor
        gated_aborted = parsed.aborted and parsed.confidence >= floor

        return AbortDecision(
            fallback=False,
            aborted=gated_aborted,
            confidence=parsed.confidence,
        )

    @staticmethod
    def __fail_open() -> AbortDecision:
        """
        Build the safe non-abort decision used when the classifier abstains.
        """

        return AbortDecision(aborted=False, confidence=0.0, fallback=True)
