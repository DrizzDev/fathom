from __future__ import annotations

from enum import StrEnum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.schemas.actions import Action


class ResolveStatus(StrEnum):
    """
    Outcome of mapping an :class:`Action`'s named target to a manifest
    element and concrete coordinates.

    Three values, distinct downstream behaviors:

    - ``RESOLVED``: a single labeled element matched cleanly; the
      returned :class:`Action` carries snapped bounds and any derived
      :class:`InputContext`.
    - ``UNRESOLVED``: no labeled element matched and no fallback located
      the target. The planner surfaces this as a tool-result so the
      next ANALYZE turn can call ``request_replan`` with category
      ``target_not_available`` or choose a different target. Maps to
      ``TARGET_UNRESOLVED`` recovery trigger when escalated.
    - ``AMBIGUOUS``: more than one candidate matched the named target
      with comparable confidence. ``candidates`` carries the top-K
      options so the next ANALYZE turn can disambiguate by label_id.
    """

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class ResolveCandidate(BaseModel):
    """
    A single candidate element returned when resolution is ambiguous.

    Carries the minimum information the planner needs to surface the
    option to the agent on the next turn without re-running geometry
    lookups against the manifest.
    """

    model_config = ConfigDict(frozen=True)

    label_id: str = Field(description="Manifest label identifier for this candidate.")
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the candidate, when available.",
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Match confidence in [0, 1] — higher means stronger match against the named target.",
    )
    bounds_preview: Optional[str] = Field(
        default=None,
        description="Compact bounds string for telemetry (e.g. 'x=,y=,w=,h='), never used for execution.",
    )


class ResolveResult(BaseModel):
    """
    Structured outcome of :class:`ReferenceResolutionService.resolve`.

    Replaces the previous contract that returned only a (possibly
    silently fuzz-matched) :class:`Action`. The status channel lets the
    EXECUTE node branch deterministically:

    - ``RESOLVED`` → execute ``action``.
    - ``UNRESOLVED`` → emit a structured failed step that the recovery
      coordinator maps to ``RecoveryTrigger.TARGET_UNRESOLVED``.
    - ``AMBIGUOUS`` → emit a structured failed step that propagates
      ``candidates`` back to the agent for disambiguation on the next
      ANALYZE turn.

    The ``action`` field always carries the resolved (or unresolved)
    action so downstream consumers don't need to special-case None.
    """

    model_config = ConfigDict(frozen=True)

    status: ResolveStatus = Field(description="Outcome of the resolution attempt.")
    action: Action = Field(
        description=(
            "The Action carried through resolution. For RESOLVED this has "
            "snapped bounds; for UNRESOLVED / AMBIGUOUS it is the original "
            "Action with no synthetic bounds invented."
        )
    )
    reason: Optional[str] = Field(
        default=None,
        description="Short diagnostic explaining why status is not RESOLVED.",
    )
    candidates: List[ResolveCandidate] = Field(
        default_factory=list,
        description=(
            "Top-K candidate elements when status is AMBIGUOUS. Empty for RESOLVED and UNRESOLVED."
        ),
    )

    @classmethod
    def resolved(cls, *, action: Action) -> "ResolveResult":
        """
        Build a clean RESOLVED outcome carrying the snapped action.
        """

        return cls(status=ResolveStatus.RESOLVED, action=action)

    @classmethod
    def unresolved(cls, *, action: Action, reason: str) -> "ResolveResult":
        """
        Build an UNRESOLVED outcome with the original action preserved.

        ``reason`` is short, mechanical text suitable for the
        ``RecoveryRequest.reason`` field — e.g. "no manifest element
        matched target 'Alright, got it button'".
        """

        return cls(status=ResolveStatus.UNRESOLVED, action=action, reason=reason)

    @classmethod
    def ambiguous(
        cls,
        *,
        action: Action,
        reason: Optional[str] = None,
        candidates: List[ResolveCandidate],
    ) -> "ResolveResult":
        """
        Build an AMBIGUOUS outcome carrying candidate options.
        """

        return cls(
            action=action,
            reason=reason,
            candidates=candidates,
            status=ResolveStatus.AMBIGUOUS,
        )

    def telemetry(self) -> Dict[str, object]:
        """
        Compact representation suitable for ``extra={}`` log payloads.
        """

        return {
            "resolve_reason": self.reason,
            "resolve_status": self.status.value,
            "candidate_count": len(self.candidates),
        }
