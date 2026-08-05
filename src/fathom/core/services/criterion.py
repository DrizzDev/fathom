"""
Model-based observation-satisfaction service for the completion gate.

The advancement policy gates an observed sub-goal on whether its exact typed observation holds on
the settled screen. That judgment is semantic, so it is made by the model via constrained structured
output (a typed :class:`CriterionAssessment`), never by a token/keyword heuristic. The host validates
only the structure of the reply and maps it to a :class:`CriterionDecision`.

A positive-only cache keyed on (workflow, sub-goal index, screen visual hash, assertion) avoids
re-asking the model about an unchanged screen; only SATISFIED verdicts are cached, so a transient
UNSATISFIED/UNCLEAR cannot lock a stuck loop against a screen the agent has actually moved past.
"""

from __future__ import annotations

from logging import getLogger
from typing import Dict, List, Tuple

from fathom.interfaces.llm import LLMPort
from fathom.schemas.criterion import (
    CriterionAssessment,
    CriterionDecision,
    CriterionSource,
    CriterionVerdict,
)
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.success import ObservationRequirement

logger = getLogger(__name__)


# Hard cap on screen elements digested into the prompt to keep token use bounded on dense screens.
LLM_SCREEN_ELEMENT_LIMIT: int = 120

# Confidence attached to a definitive model verdict so it clears the advancement confidence floor;
# UNCLEAR stays below it so ambiguity never advances.
DEFINITIVE_CONFIDENCE: float = 1.0
UNCLEAR_CONFIDENCE: float = 0.5


_SYSTEM_INSTRUCTION: str = """\
You judge whether an observation about a mobile UI screen is satisfied on the CURRENT screen.
Apply these rules strictly:

1. Only the visible post-state counts. Do NOT infer that an earlier action happened. If the
   observation says "X is selected"/"X is populated"/"screen Y is open", you need observable
   evidence of that state (transition, selected styling, populated value, the screen's own
   content), not merely that X is present.
2. verdict=satisfied only when the screen clearly shows the observed state holds.
3. verdict=unsatisfied only when the screen plainly contradicts it or the state is conspicuously
   absent.
4. verdict=unclear when the screen does not give enough evidence either way. Do not guess.
"""


class CriterionObserver:
    """
    Adjudicates whether a typed observation holds on the settled screen, via the model.

    The single tier is a constrained structured-output call returning a typed
    :class:`CriterionAssessment`. Failures degrade to UNCLEAR (never a false satisfaction). Only
    SATISFIED verdicts are cached, keyed on the screen visual hash and the exact assertion.
    """

    def __init__(self, *, llm: LLMPort) -> None:
        """
        Bind the observer to the LLM port that adjudicates observations.
        """

        self.__llm = llm
        self.__cache: Dict[Tuple[str, int, str, str], CriterionDecision] = {}

    async def check(
        self,
        *,
        workflow_id: str,
        index: int,
        requirement: ObservationRequirement,
        observation: ScreenObservation,
    ) -> CriterionDecision:
        """
        Evaluate whether the exact typed observation requirement is observable on the screen.
        """

        criterion = requirement.assertion.strip()
        cache_key = (workflow_id, index, observation.hashes.visual_hash, criterion)

        cached = self.__cache.get(cache_key)
        if cached is not None:
            return CriterionDecision(
                verdict=cached.verdict,
                source=CriterionSource.CACHE,
                confidence=cached.confidence,
                evidence=cached.evidence,
                notes=cached.notes,
            )

        decision = await self.__adjudicate(criterion=criterion, observation=observation)
        self.__cache_if_satisfied(cache_key=cache_key, decision=decision)
        return decision

    async def __adjudicate(
        self, *, criterion: str, observation: ScreenObservation
    ) -> CriterionDecision:
        """
        Ask the model, under constrained decoding, whether the observation holds; degrade to UNCLEAR.
        """

        prompt = self.__build_prompt(
            criterion=criterion, screen_digest=self.__compact_screen(observation=observation)
        )

        try:
            result = await self.__llm.generate(
                use_cache=False,
                prompt=(prompt,),
                system_instruction=_SYSTEM_INSTRUCTION,
                structured_output=StructuredOutput(payload=CriterionAssessment),
            )
            assessment = CriterionAssessment.model_validate_json(result.content or "")
        except Exception as exception:  # noqa: BLE001 - broad catch keeps the loop alive
            logger.warning(
                "criterion.llm.failed",
                extra={
                    "event": "criterion.llm.failed",
                    "error.message": str(exception),
                    "error.type": type(exception).__name__,
                    "component": "core.services.criterion",
                },
            )
            return CriterionDecision(
                evidence=(),
                confidence=0.0,
                source=CriterionSource.LLM,
                verdict=CriterionVerdict.UNCLEAR,
                notes=f"criterion adjudication raised: {type(exception).__name__}",
            )

        confidence = (
            UNCLEAR_CONFIDENCE
            if assessment.verdict is CriterionVerdict.UNCLEAR
            else DEFINITIVE_CONFIDENCE
        )
        return CriterionDecision(
            notes=None,
            confidence=confidence,
            verdict=assessment.verdict,
            source=CriterionSource.LLM,
            evidence=(assessment.reason,),
        )

    def __cache_if_satisfied(
        self, *, decision: CriterionDecision, cache_key: Tuple[str, int, str, str]
    ) -> None:
        """
        Cache only SATISFIED verdicts so a transient UNSATISFIED/UNCLEAR cannot lock a stuck loop.
        """

        if decision.verdict is CriterionVerdict.SATISFIED:
            self.__cache[cache_key] = decision

    @staticmethod
    def __compact_screen(*, observation: ScreenObservation) -> str:
        """
        Compact textual digest of the screen for the adjudication prompt.
        """

        lines: List[str] = [f"activity: {observation.activity}"]

        for element in observation.elements[:LLM_SCREEN_ELEMENT_LIMIT]:
            text = (element.text or "").strip()
            if text:
                lines.append(f"- {text}")

        if observation.keyboard.visibility.name != "HIDDEN":
            lines.append(f"keyboard: {observation.keyboard.visibility.value}")

        for overlay in observation.overlays:
            description = getattr(overlay, "description", None) or "overlay"
            lines.append(f"overlay: {description}")

        return "\n".join(lines)

    @staticmethod
    def __build_prompt(*, criterion: str, screen_digest: str) -> str:
        """
        Construct the user-turn prompt for the adjudication call.
        """

        return (
            "Determine whether this observation holds on the current screen.\n\n"
            f"OBSERVATION:\n{criterion}\n\n"
            f"CURRENT SCREEN:\n{screen_digest}"
        )
