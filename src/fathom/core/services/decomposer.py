"""Intent decomposition service for sequential sub-goal execution."""

import json
import logging
from typing import List, Optional

from pydantic import BaseModel, ValidationError, field_validator

from fathom.interfaces.llm import LLMPort
from fathom.schemas.subgoal import SubGoal, SubGoalStatus

logger = logging.getLogger(__name__)


class DecompositionSchema(BaseModel):
    """Pydantic schema for strict intent decomposition validation."""

    sub_goals: List[str]
    confidence: Optional[float] = 0.9

    @field_validator("sub_goals")
    @classmethod
    def validate_sub_goals(cls, v: List[str]) -> List[str]:
        """Validate that sub_goals is a non-empty list of non-empty strings."""
        if not v:
            raise ValueError("sub_goals must not be empty")
        if len(v) > 50:
            raise ValueError("sub_goals must not exceed 50 items")
        return [goal.strip() for goal in v if goal.strip()]


class IntentDecomposer:
    """
    Decomposes a high-level intent into sequential sub-goals using LLM.

    Sub-goals must be:
    - Atomic (one action per goal)
    - Sequential (ordered, no skipping)
    - Complete (sufficient to achieve parent intent)
    - Executable (within agent capabilities)
    """

    DECOMPOSITION_PROMPT = """You are an expert at breaking down user intents into executable micro-tasks.

INTENT: {intent}

INSTRUCTIONS:
1. Break down the intent into sequential, non-skippable steps
2. Each step must be atomic and testable
3. Steps must be in execution order (no parallelization)
4. Each step should be 1-2 sentences, action-oriented
5. CRITICAL: Do not skip any steps required to achieve the intent
6. CRITICAL: You MUST use the user's exact wording and terminology wherever possible
   - Do NOT paraphrase, generalize, or rephrase their specific requests
   - Preserve specific app names, button names, field names, and action verbs
   - Keep technical terms and product names exactly as stated

EXAMPLES:
✓ GOOD: User says "Tap the login button" → Sub-goal: "Tap the login button"
✗ BAD: User says "Tap the login button" → Sub-goal: "Authenticate with credentials"

✓ GOOD: User says "Enter password 'test123'" → Sub-goal: "Enter password 'test123'"
✗ BAD: User says "Enter password 'test123'" → Sub-goal: "Input user credentials"

✓ GOOD: User says "Open Settings app" → Sub-goal: "Open Settings app"
✗ BAD: User says "Open Settings app" → Sub-goal: "Navigate to system configuration"

Return ONLY a valid JSON with this structure:
{{
    "sub_goals": ["step 1", "step 2", "step 3"],
    "confidence": 0.9
}}

Return ONLY the JSON, no other text."""

    def __init__(self, llm: LLMPort) -> None:
        """
        Initialize decomposer with LLM service.

        Args:
            llm: LLM interface for generating decompositions
        """
        self.__llm = llm

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

        # Single-step bypass: if intent is simple/atomic, skip decomposition
        # to preserve exact user wording
        if self.__is_single_step_intent(intent):
            logger.info("[Decomposer] Single-step intent detected, bypassing decomposition")
            return self.__fallback_decomposition(intent)

        # Prepare prompt
        prompt = self.DECOMPOSITION_PROMPT.format(intent=intent)

        # Call LLM to decompose
        try:
            result = await self.__llm.generate(
                use_cache=False,
                prompt=[prompt],
                system_instruction="You are an expert task planner. Always preserve the user's exact wording.",
            )
            response = result.content
        except Exception as e:
            logger.warning(f"[Decomposer] LLM decomposition failed: {e}, using fallback")
            return self.__fallback_decomposition(intent)

        # Parse response strict Schema validation
        try:
            parsed = json.loads(response)
            schema = DecompositionSchema(**parsed)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                f"[Decomposer] Failed to parse response (schema validation failed: {e}), using fallback"
            )
            return self.__fallback_decomposition(intent)

        # Check confidence threshold
        if schema.confidence and schema.confidence < 0.5:
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

    def __is_single_step_intent(self, intent: str) -> bool:
        """
        Heuristic to detect single-step intents that don't need decomposition.

        Args:
            intent: Intent string to check

        Returns:
            True if intent appears to be a single atomic action
        """
        # Simple heuristics:
        # - No conjunctions (and, then, after, before)
        # - No commas (except within quotes)
        # - Short length (< 80 chars suggests single action)
        # - Single verb pattern

        intent_lower = intent.lower()

        # Check for multi-step indicators
        multi_step_indicators = [
            " and ",
            " then ",
            " after ",
            " before ",
            " next ",
            " finally ",
            ", and",
            ", then",
            ", after",
            ", next",
        ]

        has_multi_step_indicator = any(
            indicator in intent_lower for indicator in multi_step_indicators
        )

        # Count commas outside quotes (simple approximation)
        comma_count = intent.count(",")

        # If short, no multi-step indicators, and few commas, likely single-step
        is_single = len(intent) < 80 and not has_multi_step_indicator and comma_count <= 1

        return is_single

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
