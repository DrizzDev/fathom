from __future__ import annotations

import time
from logging import getLogger
from typing import Tuple

from pydantic import ValidationError

from fathom.core.exceptions import LanguageComplianceError
from fathom.core.prompts.generation import FlowPromptBuilder
from fathom.interfaces.generator import FlowGenerator
from fathom.interfaces.llm import LLMPort
from fathom.schemas.flow import Evidence, Flow, Issue
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import GenerateResult

logger = getLogger(__name__)


class LlmFlowGenerator(FlowGenerator):
    """
    Generates a flow by prompting an LLM with Flow-typed constrained output.
    """

    def __init__(self, *, llm: LLMPort, prompt: FlowPromptBuilder, use_cache: bool) -> None:
        """
        Bind the LLM port, the prompt builder, and the cache preference.
        """

        self.__llm = llm
        self.__prompt = prompt
        self.__use_cache = use_cache
        self.__output = StructuredOutput(payload=Flow)

    async def generate(self, *, evidence: Evidence, feedback: Tuple[Issue, ...] = ()) -> Flow:
        """
        Prompt the LLM for a flow and validate the response at the boundary.
        """

        started = time.perf_counter()

        logger.info(
            "script llm request started",
            extra={
                "event": "script.llm.request.started",
                "script.model": self.__llm.model_name,
                "script.cache_requested": self.__use_cache,
                "script.feedback_count": len(feedback),
                "script.structured_output": bool(self.__output),
            },
        )

        try:
            result = await self.__llm.generate(
                use_cache=self.__use_cache,
                structured_output=self.__output,
                system_instruction=self.__prompt.system_instruction(),
                prompt=[self.__prompt.user_prompt(evidence=evidence, feedback=feedback)],
            )
        except Exception as exception:
            logger.warning(
                "script llm request failed",
                extra={
                    "event": "script.llm.request.failed",
                    "script.model": self.__llm.model_name,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                    "duration.ms": round((time.perf_counter() - started) * 1000, 3),
                },
            )
            raise

        self.__log_completed(result=result, started=started)
        return self.__parse(content=result.content)

    def __log_completed(self, *, result: GenerateResult, started: float) -> None:
        """
        Record the LLM response metrics at the boundary, never the payload.
        """

        metrics = result.metrics

        logger.info(
            "script llm request completed",
            extra={
                "event": "script.llm.request.completed",
                "script.model": self.__llm.model_name,
                "script.prompt_tokens": metrics.get("prompt_tokens"),
                "script.cached_tokens": metrics.get("cached_tokens"),
                "script.output_tokens": metrics.get("completion_tokens"),
                "duration.ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )

    def __parse(self, *, content: str) -> Flow:
        """
        Validate the LLM response as a Flow, failing explicitly when it does not conform.
        """

        try:
            return Flow.model_validate_json(content)
        except ValidationError as exception:
            raise LanguageComplianceError(
                f"LLM returned a non-conforming flow: {exception}"
            ) from exception
