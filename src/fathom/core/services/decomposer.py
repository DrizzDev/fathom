"""Intent decomposition service for sequential sub-goal execution."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from pydantic import ValidationError

from fathom.constants.reasoning import MINIMUM_DECOMPOSITION_CONFIDENCE
from fathom.core.exceptions import ConfigurationError, DecompositionError, TranslationError
from fathom.core.prompts.factory import PromptFactory
from fathom.core.services.translation import ProposalTranslator
from fathom.interfaces.llm import LLMPort, PromptPart
from fathom.schemas.configuration import LLMConfiguration
from fathom.schemas.decomposition import DecomposedTask, DecompositionSchema
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.subgoal import SubGoal


class DecompositionAugmentation:
    """
    Optional caller-supplied hooks that enrich a decomposition prompt with extra context. All hooks
    default to empty so subclasses override only what they need.
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
    Decomposes an intent into one accepted plan of sequential sub-goals via the LLM.

    Runs exactly one DecompositionPhase: one initial structured generation and at
    most one validation-repair. There is no fallback: an unrecoverable phase fails
    closed with :class:`DecompositionError` so the run executes nothing.
    """

    def __init__(self, llm: LLMPort, *, translator: ProposalTranslator) -> None:
        self.__llm = llm
        self.__translator = translator
        self.__configuration = LLMConfiguration()
        self.__prompt_builder = PromptFactory.get_decomposition_builder(model_name=llm.model_name)

    @classmethod
    def with_configuration(
        cls, *, llm: LLMPort, translator: ProposalTranslator, configuration: LLMConfiguration
    ) -> "IntentDecomposer":
        """
        Build decomposer with an explicit LLM configuration (caching, etc.).
        """

        decomposer = cls(llm=llm, translator=translator)
        decomposer.__configuration = configuration

        return decomposer

    async def decompose(
        self,
        intent: str,
        *,
        augmentation: Optional[DecompositionAugmentation] = None,
    ) -> List[SubGoal]:
        """
        Decompose intent into one accepted plan of sub-goals, or fail closed.
        """

        if not intent or not intent.strip():
            raise ConfigurationError("Intent cannot be empty")

        parts, system_instruction = self.__build_prompt(intent=intent, augmentation=augmentation)

        content = await self.__generate(
            parts=parts, system_instruction=system_instruction, intent=intent
        )
        try:
            schema = self.__accept(content=content)
        except (ValueError, ValidationError) as finding:
            schema = await self.__repair(
                intent=intent,
                parts=parts,
                finding=str(finding),
                system_instruction=system_instruction,
            )

        return self.__build_sub_goals(schema=schema, intent=intent)

    def __build_prompt(
        self, *, intent: str, augmentation: Optional[DecompositionAugmentation]
    ) -> Tuple[List[PromptPart], str]:
        """
        Assemble the decomposition prompt parts and system instruction, applying any augmentation.
        """

        user_prompt = self.__prompt_builder.build_user_prompt(intent=intent)
        system_instruction = self.__prompt_builder.build_system_instruction()

        extra_parts: Sequence[PromptPart] = ()
        if augmentation is not None:
            extra_parts = augmentation.extra_prompt_parts()
            user_prompt = f"{augmentation.user_preamble()}{user_prompt}"
            system_instruction = f"{system_instruction}{augmentation.system_addendum()}"

        return [user_prompt, *extra_parts], system_instruction

    async def __generate(
        self, *, parts: List[PromptPart], system_instruction: str, intent: str
    ) -> str:
        """
        Run one structured-output generation, wrapping any provider failure as a fail-closed error.
        """

        try:
            result = await self.__llm.generate(
                prompt=parts,
                system_instruction=system_instruction,
                use_cache=self.__configuration.use_cache,
                structured_output=StructuredOutput(payload=DecompositionSchema),
            )
        except Exception as exception:
            raise DecompositionError(
                intent=intent, reason=f"generation failed: {exception}"
            ) from exception

        return result.content

    @staticmethod
    def __accept(*, content: str) -> DecompositionSchema:
        """
        Validate raw output into a typed plan and enforce the confidence floor.
        """

        schema = DecompositionSchema.model_validate_json(content)
        confidence = schema.confidence

        if confidence is not None and confidence < MINIMUM_DECOMPOSITION_CONFIDENCE:
            raise ValueError(
                f"confidence {confidence:.2f} below floor {MINIMUM_DECOMPOSITION_CONFIDENCE:.2f}"
            )

        return schema

    async def __repair(
        self, *, intent: str, parts: List[PromptPart], system_instruction: str, finding: str
    ) -> DecompositionSchema:
        """
        Run the single bounded validation-repair; fail closed when the retry is still invalid.
        """

        repair_parts: List[PromptPart] = [*parts, self.__repair_addendum(finding=finding)]
        content = await self.__generate(
            parts=repair_parts, system_instruction=system_instruction, intent=intent
        )

        try:
            return self.__accept(content=content)
        except (ValueError, ValidationError) as exception:
            raise DecompositionError(
                intent=intent, reason=f"plan invalid after repair: {exception}"
            ) from exception

    @staticmethod
    def __repair_addendum(*, finding: str) -> str:
        """
        Compose the corrective instruction appended for the single validation-repair attempt.
        """

        return (
            f"\n\nYour previous response was rejected: {finding}\n"
            "Return ONLY corrected JSON conforming exactly to the required schema. Every sub_goal "
            "must include an objective and one typed proposal (observed, command, or capture)."
        )

    def __build_sub_goals(self, *, schema: DecompositionSchema, intent: str) -> List[SubGoal]:
        """
        Materialize the accepted typed plan into sequential sub-goals, translating each proposal.
        """

        return [
            self.__build_sub_goal(index=index, task=task, intent=intent)
            for index, task in enumerate(schema.sub_goals)
        ]

    def __build_sub_goal(self, *, index: int, task: DecomposedTask, intent: str) -> SubGoal:
        """
        Translate one validated proposal into canonical success and build the sub-goal, or fail closed.
        """

        try:
            success = self.__translator.translate(intent=intent, proposal=task.proposal)
        except TranslationError as exception:
            raise DecompositionError(
                intent=intent,
                reason=f"proposal translation failed for sub-goal {index}: {exception}",
            ) from exception

        return SubGoal(index=index, objective=task.objective, success=success)
