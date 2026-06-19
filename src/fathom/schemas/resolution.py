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
      next ANALYZE turn can choose a different target or ask the user.
      Maps to ``TARGET_UNRESOLVED`` recovery trigger when escalated.
    - ``AMBIGUOUS``: more than one candidate matched the named target
      with comparable confidence. ``candidates`` carries the top-K
      options so the next ANALYZE turn can disambiguate by label_id.
    """

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class UnresolvedKind(StrEnum):
    """
    Machine-readable taxonomy of why a resolution attempt did not resolve.

    Distinct from the free-text ``reason`` field on :class:`ResolveResult`
    so downstream consumers can branch on the failure cause without
    string matching.
    """

    OTHER = "other"
    AXIS_MISMATCH = "axis_mismatch"
    INVALID_BOUNDS = "invalid_bounds"
    MISSING_BOUNDS = "missing_bounds"
    EMPTY_MANIFEST = "empty_manifest"
    LABEL_NOT_FOUND = "label_not_found"
    GENERIC_CONTAINER = "generic_container"


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
    - ``UNRESOLVED`` → emit a structured failed step that routes back
      through the normal planner loop.
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
    unresolved_kind: Optional[UnresolvedKind] = Field(
        default=None,
        description=(
            "Machine-readable kind explaining why status is UNRESOLVED. "
            "None for RESOLVED and AMBIGUOUS outcomes."
        ),
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
    def unresolved(
        cls,
        *,
        reason: str,
        action: Action,
        kind: UnresolvedKind = UnresolvedKind.OTHER,
    ) -> "ResolveResult":
        """
        Build an UNRESOLVED outcome with the original action preserved.

        ``reason`` is short, mechanical text, e.g. "no manifest element matched target 'Alright, got it button'".
        ``kind`` carries the machine-readable failure taxonomy used by downstream routing.
        """

        return cls(
            action=action,
            reason=reason,
            unresolved_kind=kind,
            status=ResolveStatus.UNRESOLVED,
        )

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
