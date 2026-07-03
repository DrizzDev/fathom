"""Intent decomposition service for sequential sub-goal execution."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from pydantic import ValidationError

from fathom.constants import ActionType
from fathom.constants.reasoning import MINIMUM_DECOMPOSITION_CONFIDENCE
from fathom.core.exceptions import ConfigurationError
from fathom.core.prompts.factory import PromptFactory
from fathom.core.services.directive import DirectivePolicy
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.decomposition import DecomposedTask, DecompositionSchema
from fathom.schemas.subgoal import (
    SubGoal,
    SubGoalStatus,
)

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
    Accepts an optional :class:`DecompositionAugmentation` to inject caller-specific context without coupling to the caller's domain.
    """

    def __init__(self, llm: LLMPort, *, directive_policy: DirectivePolicy) -> None:
        self.__llm = llm
        self.__directive_policy = directive_policy
        self.__configuration = LLMConfiguration()
        self.__prompt_builder = PromptFactory.get_decomposition_builder(model_name=llm.model_name)

    @classmethod
    def with_configuration(
        cls, *, llm: LLMPort, directive_policy: DirectivePolicy, configuration: LLMConfiguration
    ) -> "IntentDecomposer":
        """
        Build decomposer with an explicit LLM configuration (caching, etc.).
        """

        decomposer = cls(llm=llm, directive_policy=directive_policy)
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
                    "event": "fallback.llm.error",
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
                    "event": "fallback.parse.error",
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

        raw_sub_goals = [
            self.__build_sub_goal(index=idx, entry=entry, confidence=schema.confidence or 0.9)
            for idx, entry in enumerate(schema.sub_goals)
        ]
        sub_goals = self.__drop_terminal_markers(sub_goals=raw_sub_goals)
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
                "sub_goals": self.__structured_dump(sub_goals=sub_goals),
            },
        )
        self.__log_each_sub_goal(sub_goals=sub_goals, augmented=augmented)
        return sub_goals

    @staticmethod
    def __drop_terminal_markers(*, sub_goals: List[SubGoal]) -> List[SubGoal]:
        """
        Strip synthetic terminal-marker sub-goals (directive=complete) emitted
        by the decomposer LLM. These add nothing actionable: VERIFY already
        adjudicates overall intent completion, and a terminal marker just
        forces an extra criterion check + LLM call on a synthetic criterion
        that the verifier rubber-stamps.

        Indices are reassigned so the surviving sub-goals stay 0-based and
        contiguous for the agent-state machine.
        """

        kept: List[SubGoal] = []
        for sub_goal in sub_goals:
            if sub_goal.directive is ActionType.COMPLETE:
                logger.info(
                    "[Decomposer] dropping terminal-marker sub-goal",
                    extra={
                        "component": "decomposer",
                        "event": "sub_goal.terminal_marker.dropped",
                        "sub_goal.original_index": sub_goal.index,
                        "sub_goal.description": sub_goal.description[:80],
                    },
                )
                continue
            kept.append(sub_goal)

        if len(kept) == len(sub_goals):
            return sub_goals

        return [sub_goal.model_copy(update={"index": idx}) for idx, sub_goal in enumerate(kept)]

    def __build_sub_goal(
        self,
        *,
        index: int,
        entry: object,
        confidence: float,
    ) -> SubGoal:
        """
        Build one :class:`SubGoal` from a normalized decomposition entry.

        Typed :class:`DecomposedTask` entries carry the structured ``directive``
        contract consumed by the completion gate. Legacy string entries (older
        prompt outputs without the directive schema) are still accepted with
        ``directive=None``; the completion gate falls back to its legacy
        signal evaluation for those.
        """

        if isinstance(entry, DecomposedTask):
            return SubGoal(
                index=index,
                confidence=confidence,
                criterion=entry.criterion,
                directive=entry.directive,
                description=entry.description,
                kind=self.__directive_policy.kind(directive=entry.directive),
            )

        description = str(entry)

        return SubGoal(
            index=index,
            confidence=confidence,
            description=description,
            kind=self.__directive_policy.kind(directive=None),
        )

    @staticmethod
    def __structured_dump(*, sub_goals: List[SubGoal]) -> List[Dict[str, Any]]:
        """
        Compact per-sub-goal payload suitable for a single structured log field.
        """

        return [
            {
                "index": sub_goal.index,
                "description": sub_goal.description,
                "directive": (sub_goal.directive.value if sub_goal.directive is not None else None),
                "criterion": sub_goal.criterion,
                "max_steps": sub_goal.max_steps,
            }
            for sub_goal in sub_goals
        ]

    @staticmethod
    def __log_each_sub_goal(*, sub_goals: List[SubGoal], augmented: bool) -> None:
        """
        Emit one structured log line per decomposed sub-goal for inspection.

        Each line is keyed by ``event="sub_goal.decomposed"`` so log filters
        can isolate the decomposition output without grepping the summary line.
        """

        for sub_goal in sub_goals:
            logger.info(
                "[Decomposer] sub-goal %d: %s -> %s",
                sub_goal.index,
                sub_goal.description[:80],
                sub_goal.directive.value if sub_goal.directive is not None else "<none>",
                extra={
                    "augmented": augmented,
                    "component": "decomposer",
                    "event": "sub_goal.decomposed",
                    "sub_goal.index": sub_goal.index,
                    "sub_goal.description": sub_goal.description,
                    "sub_goal.directive": (
                        sub_goal.directive.value if sub_goal.directive is not None else None
                    ),
                    "sub_goal.criterion": sub_goal.criterion,
                    "sub_goal.max_steps": sub_goal.max_steps,
                    "sub_goal.confidence": sub_goal.confidence,
                },
            )

    @staticmethod
    def __fallback(*, intent: str) -> List[SubGoal]:
        """
        Single-goal fallback when LLM call or parsing fails.
        """

        logger.info("[Decomposer] Using fallback single-step decomposition")
        return [
            SubGoal(
                index=0,
                description=intent,
                status=SubGoalStatus.PENDING,
                confidence=0.5,
            )
        ]
