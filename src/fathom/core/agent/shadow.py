from __future__ import annotations

from typing import List, Optional, Tuple

from fathom.constants.assessment import ShadowDivergenceKind, VisualVerdict
from fathom.core.agent.eligibility import Eligibility
from fathom.schemas.assessment import ShadowDivergence, VisualAssessment
from fathom.schemas.success import Success
from fathom.schemas.target import TargetAuthority


class ShadowAssessor:
    """
    Records how a shadow visual assessment diverges from live behavior and the goal's evidence source.

    This is observe-and-record only: the live action still executes and no goal advances from this.
    The host — not the model schema — enforces which goals may carry a visual assessment and that a bound
    target's foreground package is present and exactly equal before a satisfied verdict may count.
    """

    def assess(
        self,
        *,
        success: Success,
        assessment: Optional[VisualAssessment],
        action_present: bool,
        assessment_malformed: bool = False,
        authority: TargetAuthority = TargetAuthority(),
        foreground_package: Optional[str] = None,
        truth_satisfied: Optional[bool] = None,
    ) -> Tuple[ShadowDivergence, ...]:
        """
        Return every way the assessment disagrees with the goal's evidence source or the live turn.
        """

        if assessment_malformed:
            return (
                ShadowDivergence(
                    kind=ShadowDivergenceKind.SCHEMA_FAILURE,
                    detail="The turn carried a visual-assessment payload that failed its schema.",
                ),
            )

        observation = Eligibility.observation(success=success)
        if observation is None:
            if assessment is not None:
                return (
                    ShadowDivergence(
                        kind=ShadowDivergenceKind.WRONG_GOAL,
                        detail="Visual assessment produced for a goal that proves completion from a receipt.",
                    ),
                )
            return ()

        if assessment is None:
            return (
                ShadowDivergence(
                    kind=ShadowDivergenceKind.MISSING_ASSESSMENT,
                    detail="A goal whose completion is proven visually produced no assessment.",
                ),
            )

        divergences: List[ShadowDivergence] = []
        satisfied = assessment.verdict is VisualVerdict.SATISFIED

        if satisfied and action_present:
            divergences.append(
                ShadowDivergence(
                    kind=ShadowDivergenceKind.SATISFIED_WITH_ACTION,
                    detail="SATISFIED verdict accompanied a proposed action on the same turn.",
                )
            )

        if satisfied and authority.bound and foreground_package != authority.package:
            divergences.append(
                ShadowDivergence(
                    kind=ShadowDivergenceKind.PACKAGE_CONTRADICTION,
                    detail=(
                        f"SATISFIED on foreground package {foreground_package!r} does not match the "
                        f"required package {authority.package!r} (a missing foreground never satisfies)."
                    ),
                )
            )

        if truth_satisfied is not None:
            if satisfied and not truth_satisfied:
                divergences.append(
                    ShadowDivergence(
                        kind=ShadowDivergenceKind.FALSE_POSITIVE,
                        detail="Verdict SATISFIED but the oracle reports the goal was not complete.",
                    )
                )
            if not satisfied and truth_satisfied:
                divergences.append(
                    ShadowDivergence(
                        kind=ShadowDivergenceKind.FALSE_NEGATIVE,
                        detail="Verdict withheld but the oracle reports the goal was complete.",
                    )
                )

        return tuple(divergences)
