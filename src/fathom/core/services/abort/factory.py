from __future__ import annotations

from typing import Optional

from fathom.core.prompts.abort import AbortPromptBuilder
from fathom.core.services.abort.composite import CompositeAbortDetector
from fathom.core.services.abort.heuristic import HeuristicAbortDetector
from fathom.core.services.abort.llm import LLMAbortDetector
from fathom.interfaces.abort import AbortDetectorPort
from fathom.interfaces.llm import LLMPort
from fathom.schemas.abort import AbortDetectorConfiguration


class AbortDetectorFactory:
    """
    Builds the composite abort detector with an LLM primary and a heuristic fallback.
    """

    @classmethod
    def build(
        cls,
        *,
        llm: LLMPort,
        prompt_builder: Optional[AbortPromptBuilder] = None,
        configuration: Optional[AbortDetectorConfiguration] = None,
    ) -> AbortDetectorPort:
        """
        Construct the composite detector pipeline used by the HITL node.
        """

        config = configuration or AbortDetectorConfiguration()

        primary = LLMAbortDetector(
            llm=llm,
            configuration=config,
            prompt_builder=prompt_builder,
        )
        fallback = HeuristicAbortDetector(configuration=config.fallback)

        return CompositeAbortDetector(primary=primary, fallback=fallback)
