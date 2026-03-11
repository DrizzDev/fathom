"""Intent decomposition service for sequential sub-goal execution."""

import json
import logging
from typing import List

from pydantic import ValidationError

from fathom.core.prompts.factory import PromptFactory
from fathom.interfaces.llm import LLMPort
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.decomposition import DecompositionSchema
from fathom.schemas.subgoal import SubGoal, SubGoalStatus

logger = logging.getLogger(__name__)


class IntentDecomposer:
    """
    Decomposes a high-level intent into sequential sub-goals using LLM.

    Sub-goals must be:
    - Atomic (one action per goal)
    - Sequential (ordered, no skipping)
    - Complete (sufficient to achieve parent intent)
    - Executable (within agent capabilities)
    """

    def __init__(self, llm: LLMPort) -> None:
        """
        Initialize decomposer with LLM service.

        Args:
            llm: LLM interface for generating decompositions
        """
        self.__llm = llm
        self.__llm_configuration = LLMConfiguration()
        self.__prompt_builder = PromptFactory.get_decomposition_builder(model_name=llm.model_name)

    @classmethod
    def with_configuration(
        cls, *, llm: LLMPort, configuration: LLMConfiguration
    ) -> "IntentDecomposer":
        """
        Build decomposer with provided LLM configuration.
        """

        decomposer = cls(llm=llm)
        decomposer.__llm_configuration = configuration
        return decomposer

    async def decompose(self, intent: str) -> List[SubGoal]:
        """
        Decompose intent into sequential sub-goals.

        Args:
            intent: High-level intent to decompose

        Returns:
            List of SubGoal objects in sequential order

        Raises:
            ValueError: If decomposition fails or produces invalid schema
        """
        if not intent or not intent.strip():
            raise ValueError("Intent cannot be empty")

        logger.info(f"[Decomposer] Starting decomposition: {intent[:100]}...")

        prompt = self.__prompt_builder.build_user_prompt(intent=intent)

        # Call LLM to decompose
        try:
            result = await self.__llm.generate(
                use_cache=False,
                prompt=[prompt],
                system_instruction=self.__prompt_builder.build_system_instruction(),
            )
            response = result.content
        except Exception as e:
            logger.warning(f"[Decomposer] LLM decomposition failed: {e}, using fallback")
            return self.__fallback_decomposition(intent)

        # Parse response with strict Schema validation
        try:
            parsed = json.loads(response)
            schema = DecompositionSchema(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                f"[Decomposer] Failed to parse response (schema validation failed: {e}), using fallback"
            )
            return self.__fallback_decomposition(intent)

        # Check confidence threshold
        if (
            schema.confidence is not None
            and schema.confidence < self.__llm_configuration.confidence_threshold
        ):
            logger.warning(f"[Decomposer] Low confidence {schema.confidence}, using fallback")
            return self.__fallback_decomposition(intent)

        # Convert to SubGoal objects
        sub_goals: List[SubGoal] = []
        for idx, description in enumerate(schema.sub_goals):
            sub_goal = SubGoal(
                index=idx, description=description, confidence=schema.confidence or 0.9
            )
            sub_goals.append(sub_goal)

        logger.info(
            f"[Decomposer] Successfully decomposed into {len(sub_goals)} sub-goals "
            f"(confidence={schema.confidence})"
        )
        return sub_goals

    def __fallback_decomposition(self, intent: str) -> List[SubGoal]:
        """
        Fallback decomposition when LLM parsing fails.
        Creates a single safe sub-goal that includes the entire intent.

        Args:
            intent: Full intent string

        Returns:
            List with single SubGoal containing the full intent
        """
        logger.info("[Decomposer] Using fallback single-step decomposition")
        return [SubGoal(index=0, description=intent, status=SubGoalStatus.PENDING, confidence=0.5)]
