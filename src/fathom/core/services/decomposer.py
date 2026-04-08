"""Intent decomposition service for sequential sub-goal execution."""

import json
import logging
from typing import Dict, List, Optional, Sequence, Union

from pydantic import ValidationError

from fathom.constants.execution import MINIMUM_DECOMPOSITION_CONFIDENCE
from fathom.core.exceptions import ConfigurationError
from fathom.core.prompts.decomposition import DECOMPOSITION_REPLAN_SCREENSHOT_NOTE
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

    async def decompose(
        self,
        intent: str,
        *,
        screenshot: Optional[bytes] = None,
        stuck_sub_goal: Optional[str] = None,
        failure_reason: Optional[str] = None,
        suggested_next_action: Optional[str] = None,
        recent_actions: Sequence[str] = (),
    ) -> List[SubGoal]:
        """
        Decompose intent into sequential sub-goals.

        Args:
            intent: High-level intent to decompose.
            screenshot: Optional current-screen screenshot. When provided
                (e.g. during replanning), the LLM can see where the agent
                currently is and plan from that state.
            stuck_sub_goal: (Replan only) description of the sub-goal the
                agent failed to complete before triggering replanning.
            failure_reason: (Replan only) the verifier's rejection reason
                or the stuck-detection trigger message.
            suggested_next_action: (Replan only) the concrete action the
                verifier suggested the model should try next, if any.
            recent_actions: (Replan only) recently-emitted action lines
                in "{kind}: {target}" form so the decomposer can avoid
                re-proposing the same dead-end path.

        Returns:
            List of SubGoal objects in sequential order.

        Raises:
            ValueError: If decomposition fails or produces invalid schema.
        """
        if not intent or not intent.strip():
            raise ConfigurationError("Intent cannot be empty")

        is_replan = bool(
            stuck_sub_goal or failure_reason or suggested_next_action or recent_actions
        )
        mode_label = "replan" if is_replan else "initial"
        logger.info(
            "[Decomposer] Starting %s decomposition: %s...",
            mode_label,
            intent[:100],
        )

        prompt_text = self.__prompt_builder.build_user_prompt(
            intent=intent,
            stuck_sub_goal=stuck_sub_goal,
            failure_reason=failure_reason,
            suggested_next_action=suggested_next_action,
            recent_actions=recent_actions,
        )

        system_instruction = self.__prompt_builder.build_system_instruction()
        if screenshot:
            system_instruction += DECOMPOSITION_REPLAN_SCREENSHOT_NOTE

        # Build prompt parts: text + optional screenshot
        prompt_parts: List[Union[str, bytes, Dict[str, str]]] = [prompt_text]
        if screenshot:
            prompt_parts.append(screenshot)

        # Call LLM to decompose
        try:
            result = await self.__llm.generate(
                use_cache=False,
                prompt=prompt_parts,
                system_instruction=system_instruction,
            )
            response = result.content
        except Exception as exception:
            logger.warning(f"[Decomposer] LLM decomposition failed: {exception}, using fallback")
            return self.__fallback_decomposition(intent)

        # Parse response with strict Schema validation
        try:
            parsed = json.loads(response)
            schema = DecompositionSchema(**parsed)
        except (json.JSONDecodeError, ValidationError) as exception:
            logger.warning(
                f"[Decomposer] Failed to parse response (schema validation failed: {exception}), using fallback"
            )
            return self.__fallback_decomposition(intent)

        # Check confidence threshold
        if schema.confidence is not None and schema.confidence < MINIMUM_DECOMPOSITION_CONFIDENCE:
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
