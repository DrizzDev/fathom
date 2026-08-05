"""
Post-action outcome service.

After an action settles, this judges whether the active sub-goal's observation requirement holds on
the settled post-action screen. The judgement is multimodal: one constrained-decoding call over the
BEFORE screenshot (marked with where the action landed) and the plain AFTER screenshot, returning a
typed :class:`VisualAssessment`. The single per-step verdict is the live completion authority for an
observed sub-goal and is reused for the shadow record. Failures degrade to UNCLEAR, never a false
SATISFIED.
"""

from __future__ import annotations

from logging import getLogger
from typing import Optional, Tuple

from fathom.constants.assessment import VisualVerdict
from fathom.core.artifact.renderer import TraceRenderer
from fathom.interfaces.llm import LLMPort
from fathom.schemas.actions import Action
from fathom.schemas.artifact import ArtifactRecord, TracePayload
from fathom.schemas.assessment import VisualAssessment
from fathom.schemas.llm import StructuredOutput
from fathom.schemas.results import TraceEmission
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.steps import StepResult
from fathom.schemas.success import ObservationRequirement

logger = getLogger(__name__)


_SYSTEM_INSTRUCTION: str = """\
You judge whether an observation about a mobile UI holds on the AFTER screen, after an action ran.

You are given two images:
- BEFORE: the pre-action screen, marked with where the action landed (a circle for a tap, an arrow
  for a swipe or scroll).
- AFTER: the plain post-action screen.

Apply these rules strictly:

1. Only the AFTER screen decides the verdict. The BEFORE mark is context for reasoning about whether
   the action had its intended effect (the tap landed off the real control, a scroll did not move
   because it started on a sticky element, a scroll from too low minimized the app, etc.).
2. verdict=SATISFIED only when the AFTER screen clearly shows the observed state holds.
3. verdict=NOT_SATISFIED only when the AFTER screen plainly contradicts it or the state is
   conspicuously absent.
4. verdict=UNCLEAR when the AFTER screen does not give enough evidence either way. Do not guess.
"""


class OutcomeObserver:
    """
    Adjudicates, via one multimodal model call, whether the observation holds on the settled screen.
    """

    def __init__(self, *, llm: LLMPort) -> None:
        """
        Bind the observer to the LLM port that adjudicates the settled screen.
        """

        self.__llm = llm

    @staticmethod
    def mark(
        *, capture: ScreenCapture, coords: Tuple[int, ...], action: Action, session: str
    ) -> bytes:
        """
        Draw only the dispatched action (tap circle or swipe arrow) on a copy of the pre-action capture.
        """

        return TraceRenderer().render(
            record=ArtifactRecord(
                created=0,
                step_number=0,
                session_id=session or "shadow",
                package_name=capture.activity or "shadow",
                payload=TracePayload(capture=capture, coords=coords, action=action),
            )
        )

    async def observe(
        self,
        *,
        session: str,
        receipt: StepResult,
        trace: Tuple[TraceEmission, ...],
        requirement: ObservationRequirement,
    ) -> Optional[VisualAssessment]:
        """
        Build the before/after frames for the settled step and adjudicate once, or None when frames are missing.
        """

        frames = self.frames(receipt=receipt, trace=trace, session=session)
        if frames is None:
            return None

        before, after = frames
        return await self.assess(requirement=requirement, before=before, after=after)

    @classmethod
    def frames(
        cls, *, receipt: StepResult, trace: Tuple[TraceEmission, ...], session: str
    ) -> Optional[Tuple[bytes, bytes]]:
        """
        Return the marked pre-action and plain post-action screen bytes for the step, or None when either is absent.
        """

        after = cls.__plain(receipt=receipt, before=False)
        before = cls.__before(receipt=receipt, trace=trace, session=session)

        if before is None or after is None:
            return None

        return before, after

    @classmethod
    def __before(
        cls, *, receipt: StepResult, trace: Tuple[TraceEmission, ...], session: str
    ) -> Optional[bytes]:
        """
        Return the pre-action screenshot marked with only the dispatched action, or the plain capture when unmarked.
        """

        for emission in trace:
            capture = emission.event.capture
            if not capture.image or len(emission.event.coords) < 2:
                continue
            try:
                return cls.mark(
                    session=session,
                    capture=capture,
                    action=receipt.step.action,
                    coords=emission.event.coords,
                )
            except Exception:  # noqa: BLE001 - a failed overlay must never fail adjudication
                break
        return cls.__plain(receipt=receipt, before=True)

    @staticmethod
    def __plain(*, receipt: StepResult, before: bool) -> Optional[bytes]:
        """
        Read the raw pre- or post-action screen bytes carried on the receipt's step artifacts.
        """

        artifacts = receipt.artifacts
        screen = artifacts.screen if artifacts is not None else None

        if screen is None:
            return None

        artifact = screen.before if before else screen.after
        return artifact.image if artifact is not None else None

    async def assess(
        self, *, requirement: ObservationRequirement, before: bytes, after: bytes
    ) -> VisualAssessment:
        """
        Ask the model whether the requirement holds on the AFTER screen; degrade to UNCLEAR on any failure.
        """

        prompt = (
            "Determine whether this observation holds on the AFTER screen.\n\n"
            f"OBSERVATION:\n{requirement.assertion.strip()}"
        )

        try:
            result = await self.__llm.generate(
                use_cache=False,
                prompt=(prompt, before, after),
                system_instruction=_SYSTEM_INSTRUCTION,
                structured_output=StructuredOutput(payload=VisualAssessment),
            )
            return VisualAssessment.model_validate_json(result.content or "")
        except Exception as exception:  # noqa: BLE001 - broad catch keeps the shadow alive
            logger.warning(
                "outcome.vision.failed",
                extra={
                    "event": "outcome.vision.failed",
                    "error.message": str(exception),
                    "error.type": type(exception).__name__,
                    "component": "core.services.outcome",
                },
            )
            return VisualAssessment(
                confidence=0.0,
                verdict=VisualVerdict.UNCLEAR,
                evidence=f"outcome adjudication raised: {type(exception).__name__}",
            )
