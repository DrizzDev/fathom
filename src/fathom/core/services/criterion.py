"""
Tri-state criterion-satisfaction service for the completion gate.

The completion gate's primary question is no longer "did the planner emit the
directive the decomposer expected?" but "is the sub-goal's observable criterion
satisfied on the current screen?" This service answers that question with a
three-tier verdict (SATISFIED / UNSATISFIED / UNCLEAR), provenance for RCA,
and a small in-memory cache keyed on the screen visual hash.

Layering:

1. Symbolic check — cheap, deterministic. Tokenises the criterion text and the
   visible screen text and computes a hit-ratio. Resolves presence/visibility
   criteria without an LLM call. Returns UNCLEAR when the ratio is below
   threshold so behavioural criteria are not falsely rejected.

2. LLM check — invoked only when the symbolic layer is UNCLEAR. The prompt
   instructs the model not to infer that earlier actions happened; only
   observable post-state counts. This is the guard against an LLM falsely
   ruling "heart icon is clicked" SATISFIED just because the icon is visible.

Decisions are cached by (workflow_id, sub_goal.index, screen_visual_hash,
criterion_text) so a stuck same-screen loop does not pay for repeated LLM
calls.
"""

from __future__ import annotations

import re
from logging import getLogger
from typing import Dict, List, Tuple

from fathom.interfaces.llm import LLMPort
from fathom.schemas.criterion import (
    CriterionDecision,
    CriterionSource,
    CriterionVerdict,
)
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.subgoal import SubGoal

logger = getLogger(__name__)


# Tokens shorter than this are too noisy (articles, fragments) to anchor a match.
SYMBOLIC_MIN_TOKEN_LEN: int = 3

# At least this fraction of significant criterion tokens must appear in the
# screen text for a symbolic SATISFIED verdict. Below this floor we return
# UNCLEAR so the LLM layer can adjudicate behavioural criteria.
SYMBOLIC_REQUIRED_HIT_RATIO: float = 0.5

# Symbolic confidence at exactly SYMBOLIC_REQUIRED_HIT_RATIO; rises with ratio.
SYMBOLIC_BASE_CONFIDENCE: float = 0.85

# Hard cap on screen elements digested into the LLM prompt to keep token use
# bounded on screens with hundreds of OCR tokens.
LLM_SCREEN_ELEMENT_LIMIT: int = 120

# Per-verdict confidence floor for LLM verdicts pending structured output.
LLM_CONFIDENCE_SATISFIED: float = 0.85
LLM_CONFIDENCE_UNSATISFIED: float = 0.8
LLM_CONFIDENCE_UNCLEAR: float = 0.5


_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "be",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "with",
        "by",
        "this",
        "that",
        "from",
        "as",
        "it",
        "its",
        "you",
        "your",
        "has",
        "have",
        "had",
        "was",
        "were",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "will",
        "shall",
    }
)


_SYSTEM_INSTRUCTION: str = """\
You judge whether a sub-goal's success criterion is observably satisfied on the
current mobile UI screen. Apply these rules strictly:

1. Only the visible post-state counts. Do NOT infer that an earlier action
   happened. If the criterion says "X is clicked", "X is selected", or "X is
   populated", you need observable evidence of the post-state (toast, screen
   transition, selected styling, keyboard appearing, sheet opening, populated
   input value, etc.) — not merely that X is present on screen.
2. Reply SATISFIED only when the screen text or elements clearly match the
   criterion's required post-state.
3. Reply UNSATISFIED only when the screen plainly contradicts the criterion
   or when the required post-state is conspicuously absent.
4. Reply UNCLEAR when the screen does not give enough evidence either way.
   Do not guess.
"""


def _tokenize(text: str) -> List[str]:
    """
    Lowercased alphanumeric tokens with short fragments and stopwords removed.

    Order is preserved (callers may inspect leading tokens), but the hit-ratio
    calculation does not depend on order.
    """

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= SYMBOLIC_MIN_TOKEN_LEN and t not in _STOPWORDS]


class CriterionChecker:
    """
    Tri-state criterion satisfaction check against a typed ScreenObservation.

    See module docstring for the symbolic-then-LLM layering and the caching
    contract. The checker never raises into the gate — LLM failures degrade to
    an UNCLEAR verdict so the completion gate's safety-net path remains in
    control of the decision.
    """

    def __init__(self, *, llm: LLMPort) -> None:
        """
        Bind the checker to an LLM port for the fallback layer.
        """

        self.__llm = llm
        self.__cache: Dict[Tuple[str, int, str, str], CriterionDecision] = {}

    async def check(
        self,
        *,
        workflow_id: str,
        sub_goal: SubGoal,
        observation: ScreenObservation,
    ) -> CriterionDecision:
        """
        Evaluate whether the sub-goal's criterion is observable on the screen.
        """

        criterion = (sub_goal.criterion or sub_goal.description or "").strip()
        if not criterion:
            return CriterionDecision(
                verdict=CriterionVerdict.UNCLEAR,
                source=CriterionSource.SYMBOLIC,
                confidence=0.0,
                evidence=(),
                notes="Sub-goal has neither criterion nor description text.",
            )

        cache_key = (
            workflow_id,
            sub_goal.index,
            observation.hashes.visual_hash,
            criterion,
        )
        cached = self.__cache.get(cache_key)
        if cached is not None:
            return CriterionDecision(
                verdict=cached.verdict,
                source=CriterionSource.CACHE,
                confidence=cached.confidence,
                evidence=cached.evidence,
                notes=cached.notes,
            )

        symbolic = self.__symbolic_check(criterion=criterion, observation=observation)
        if symbolic.verdict is not CriterionVerdict.UNCLEAR:
            self.__cache[cache_key] = symbolic
            return symbolic

        decision = await self.__llm_check(criterion=criterion, observation=observation)
        self.__cache[cache_key] = decision
        return decision

    @staticmethod
    def __symbolic_check(
        *, criterion: str, observation: ScreenObservation
    ) -> CriterionDecision:
        """
        Token-overlap check between criterion text and visible screen text.
        """

        criterion_tokens = _tokenize(criterion)
        if not criterion_tokens:
            return CriterionDecision(
                verdict=CriterionVerdict.UNCLEAR,
                source=CriterionSource.SYMBOLIC,
                confidence=0.0,
                evidence=(),
                notes="Criterion has no significant tokens after stopword removal.",
            )

        screen_text = " ".join(
            (element.text or "") for element in observation.elements if element.text
        )
        screen_tokens = set(_tokenize(screen_text))

        matched = [token for token in criterion_tokens if token in screen_tokens]
        ratio = len(matched) / len(criterion_tokens)

        if ratio >= SYMBOLIC_REQUIRED_HIT_RATIO:
            # Confidence scales with how strongly the tokens hit. A perfect
            # hit yields up to SYMBOLIC_BASE_CONFIDENCE + 0.15 ≈ 1.0; the
            # required floor yields exactly SYMBOLIC_BASE_CONFIDENCE.
            confidence = min(
                1.0,
                SYMBOLIC_BASE_CONFIDENCE
                + 0.15 * (ratio - SYMBOLIC_REQUIRED_HIT_RATIO),
            )
            return CriterionDecision(
                verdict=CriterionVerdict.SATISFIED,
                source=CriterionSource.SYMBOLIC,
                confidence=confidence,
                evidence=tuple(matched[:6]),
                notes=f"{len(matched)}/{len(criterion_tokens)} tokens matched.",
            )

        return CriterionDecision(
            verdict=CriterionVerdict.UNCLEAR,
            source=CriterionSource.SYMBOLIC,
            confidence=0.0,
            evidence=tuple(matched[:6]),
            notes=(
                f"Only {len(matched)}/{len(criterion_tokens)} tokens matched "
                f"(threshold {SYMBOLIC_REQUIRED_HIT_RATIO})."
            ),
        )

    async def __llm_check(
        self, *, criterion: str, observation: ScreenObservation
    ) -> CriterionDecision:
        """
        LLM-backed adjudication for criteria the symbolic layer could not resolve.

        Any exception from the LLM port degrades to an UNCLEAR verdict so the
        gate's safety-net path takes the decision; this service must not crash
        the agent loop.
        """

        screen_digest = self.__compact_screen(observation=observation)
        prompt = self.__build_prompt(criterion=criterion, screen_digest=screen_digest)
        try:
            result = await self.__llm.generate(
                use_cache=False,
                prompt=(prompt,),
                system_instruction=_SYSTEM_INSTRUCTION,
            )
        except Exception as exc:  # noqa: BLE001 - intentional broad catch; see docstring
            logger.warning(
                "criterion.llm.failed",
                extra={
                    "component": "core.services.criterion",
                    "event": "criterion.llm.failed",
                    "error.type": type(exc).__name__,
                    "error.message": str(exc),
                },
            )
            return CriterionDecision(
                verdict=CriterionVerdict.UNCLEAR,
                source=CriterionSource.LLM,
                confidence=0.0,
                evidence=(),
                notes=f"LLM check raised: {type(exc).__name__}",
            )

        return self.__parse_llm_verdict(content=result.content or "")

    @staticmethod
    def __compact_screen(*, observation: ScreenObservation) -> str:
        """
        Compact textual digest of the screen for the LLM prompt.
        """

        lines: List[str] = [f"activity: {observation.activity}"]
        for element in observation.elements[:LLM_SCREEN_ELEMENT_LIMIT]:
            text = (element.text or "").strip()
            if not text:
                continue
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
        Construct the user-turn prompt for the LLM check.
        """

        return (
            "Determine whether the criterion is satisfied by the current screen.\n\n"
            f"CRITERION:\n{criterion}\n\n"
            f"CURRENT SCREEN:\n{screen_digest}\n\n"
            "Reply with exactly one of: SATISFIED, UNSATISFIED, UNCLEAR.\n"
            "Then add one short sentence of reasoning."
        )

    @staticmethod
    def __parse_llm_verdict(*, content: str) -> CriterionDecision:
        """
        Map the LLM's free-text reply onto the tri-state verdict.
        """

        stripped = content.strip()
        head = stripped.upper()
        if head.startswith("SATISFIED"):
            verdict = CriterionVerdict.SATISFIED
            confidence = LLM_CONFIDENCE_SATISFIED
        elif head.startswith("UNSATISFIED"):
            verdict = CriterionVerdict.UNSATISFIED
            confidence = LLM_CONFIDENCE_UNSATISFIED
        else:
            verdict = CriterionVerdict.UNCLEAR
            confidence = LLM_CONFIDENCE_UNCLEAR

        # Keep evidence bounded: the leading line is the verdict + rationale,
        # which is the only piece a human RCA reader benefits from.
        leading = stripped.splitlines()[0][:240] if stripped else ""
        return CriterionDecision(
            verdict=verdict,
            source=CriterionSource.LLM,
            confidence=confidence,
            evidence=(leading,) if leading else (),
            notes=None,
        )
