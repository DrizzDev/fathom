"""Intent decomposition service for sequential sub-goal execution."""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Sequence

from pydantic import ValidationError

from fathom.constants.reasoning import MINIMUM_DECOMPOSITION_CONFIDENCE
from fathom.core.exceptions import ConfigurationError
from fathom.core.prompts.factory import PromptFactory
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.decomposition import DecompositionSchema
from fathom.schemas.subgoal import SubGoal, SubGoalStatus

logger = logging.getLogger(__name__)


class DecompositionAugmentation:
    """
    Optional caller-supplied hooks that enrich a decomposition prompt with
    extra context. All hooks default to empty so subclasses override only what they need.
    """

    def system_addendum(self) -> str:
        """
        Text appended to the base system instruction.
        """

        return ""

    def user_preamble(self) -> str:
        """
        Text prepended to the base user prompt.
        """

        return ""

    def extra_prompt_parts(self) -> Sequence[PromptPart]:
        """
        Extra prompt parts (e.g. images) appended after the user prompt.
        """

        return ()


class IntentDecomposer:
    """
    Decomposes a high-level intent into sequential sub-goals via the LLM.
    Accepts an optional :class:`DecompositionAugmentation` to inject
    caller-specific context without coupling to the caller's domain.
    """

    def __init__(self, llm: LLMPort) -> None:
        self.__llm = llm
        self.__configuration = LLMConfiguration()
        self.__prompt_builder = PromptFactory.get_decomposition_builder(model_name=llm.model_name)

    @classmethod
    def with_configuration(
        cls, *, llm: LLMPort, configuration: LLMConfiguration
    ) -> "IntentDecomposer":
        """
        Build decomposer with an explicit LLM configuration (caching, etc.).
        """

        decomposer = cls(llm=llm)
        decomposer.__configuration = configuration

        return decomposer

    async def decompose(
        self,
        intent: str,
        *,
        augmentation: Optional[DecompositionAugmentation] = None,
    ) -> List[SubGoal]:
        """
        Decompose intent into sequential sub-goals.
        """

        if not intent or not intent.strip():
            raise ConfigurationError("Intent cannot be empty")

        augmented = augmentation is not None
        logger.info(
            "[Decomposer] decomposing intent (augmented=%s): %s",
            augmented,
            intent[:100],
            extra={
                "event": "start",
                "augmented": augmented,
                "component": "decomposer",
                "intent_preview": intent[:160],
            },
        )

        extra_parts: Sequence[PromptPart] = ()
        user_prompt = self.__prompt_builder.build_user_prompt(intent=intent)
        system_instruction = self.__prompt_builder.build_system_instruction()

        if augmentation is not None:
            extra_parts = augmentation.extra_prompt_parts()
            user_prompt = f"{augmentation.user_preamble()}{user_prompt}"
            system_instruction = f"{system_instruction}{augmentation.system_addendum()}"

        prompt_parts: List[PromptPart] = [user_prompt, *extra_parts]

        try:
            result = await self.__llm.generate(
                prompt=prompt_parts,
                system_instruction=system_instruction,
                use_cache=self.__configuration.use_cache,
            )
            response = result.content
        except Exception as exception:
            logger.warning(
                "[Decomposer] LLM call failed (%s) — using fallback",
                exception,
                extra={
                    "error": str(exception),
                    "component": "decomposer",
                    "event": "fallback_llm_error",
                },
            )
            return self.__fallback(intent=intent)

        try:
            schema = DecompositionSchema(**json.loads(response))
        except (json.JSONDecodeError, ValidationError) as exception:
            logger.warning(
                "[Decomposer] parse failed (%s) — using fallback",
                exception,
                extra={
                    "error": str(exception),
                    "component": "decomposer",
                    "event": "fallback_parse_error",
                    "response_preview": response[:200] if response else "",
                },
            )
            return self.__fallback(intent=intent)

        if schema.confidence is not None and schema.confidence < MINIMUM_DECOMPOSITION_CONFIDENCE:
            logger.warning(
                "[Decomposer] confidence %.2f below floor %.2f — using fallback",
                schema.confidence,
                MINIMUM_DECOMPOSITION_CONFIDENCE,
                extra={
                    "component": "decomposer",
                    "event": "low_confidence",
                    "confidence": schema.confidence,
                    "floor": MINIMUM_DECOMPOSITION_CONFIDENCE,
                },
            )
            return self.__fallback(intent=intent)

        sub_goals = [
            SubGoal(index=idx, description=description, confidence=schema.confidence or 0.9)
            for idx, description in enumerate(schema.sub_goals)
        ]
        logger.info(
            "[Decomposer] produced %d sub-goal(s) confidence=%s",
            len(sub_goals),
            schema.confidence,
            extra={
                "event": "success",
                "augmented": augmented,
                "component": "decomposer",
                "sub_goal_count": len(sub_goals),
                "confidence": schema.confidence,
            },
        )
        return sub_goals

    @staticmethod
    def __fallback(*, intent: str) -> List[SubGoal]:
        """
        Single-goal fallback when LLM call or parsing fails.
        """

        logger.info("[Decomposer] Using fallback single-step decomposition")
        return [SubGoal(index=0, description=intent, status=SubGoalStatus.PENDING, confidence=0.5)]
