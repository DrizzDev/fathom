from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants.screen import (
    ACTION_EFFECT_NO_PROGRESS_CONTENT_DIFF_BELOW_OR_EQ,
    ACTION_EFFECT_NO_PROGRESS_PHASH_BELOW_OR_EQ,
    ACTION_EFFECT_NO_PROGRESS_SCROLL_DISTANCE_PX_BELOW_OR_EQ,
    ACTION_EFFECT_NO_PROGRESS_SSIM_ABOVE,
    ACTION_EFFECT_PROGRESS_CONTENT_DIFF_ABOVE,
    ACTION_EFFECT_PROGRESS_PHASH_ABOVE,
    ACTION_EFFECT_PROGRESS_SCROLL_DISTANCE_PX_ABOVE,
    ACTION_EFFECT_PROGRESS_SSIM_BELOW,
)
from fathom.schemas.base.common import SealedModel
from fathom.schemas.screens import ScreenDiff


class ActionEffectSignalCounts(BaseModel):
    """
    Per-classifier signal tally that produced :class:`ActionEffectStatus`.
    Diagnostic-only — not consumed by the agent prompt.
    """

    model_config = ConfigDict(frozen=True)

    progress: int = Field(
        ge=0,
        default=0,
        description="Number of metrics whose value crossed a progress threshold.",
    )
    no_progress: int = Field(
        ge=0,
        default=0,
        description="Number of metrics whose value satisfied a no-progress threshold.",
    )
    expected: int = Field(
        ge=1,
        default=1,
        description="Number of signals available on the diff (pHash is always available).",
    )


class ActionEffectStatus(StrEnum):
    """
    Coarse classification of what an action did to the screen.

    - ``PROGRESS``: the screen visibly moved forward (significant pHash
      jump, low SSIM, large content diff, OR meaningful scroll).
    - ``NO_PROGRESS``: every available signal indicates the screen did
      not meaningfully change after the action.
    - ``UNCERTAIN``: signals are mixed; some indicate change, others do
      not. The agent should treat this as "ambiguous outcome" rather
      than "no effect".
    - ``REGRESSION``: the foreground left the target application; produced
      only by the direction-aware trial classifier, never by the live path.
    """

    PROGRESS = "progress"
    UNCERTAIN = "uncertain"
    REGRESSION = "regression"
    NO_PROGRESS = "no_progress"


class ActionEffect(BaseModel):
    """
    Structured outcome of one executed action.

    Carries the deterministic ``status`` classifier — the load-bearing signal for the prompt — plus the
    raw metrics that produced it, which are diagnostic context, not the contract. The classifier
    thresholds (see ``constants/screen.py``) are fixture-pinned against real run traces.
    """

    model_config = ConfigDict(frozen=True)

    status: ActionEffectStatus = Field(
        description="Coarse outcome classification consumed by the prompt and the loop detector."
    )

    visual_progress: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Normalized visual change in [0, 1]. Derived from ``1 - SSIM`` "
            "when SSIM is available, otherwise from the pHash hamming distance normalized by max possible distance."
        ),
    )
    phash_distance: int = Field(
        ge=0,
        description="Raw pHash hamming distance, retained for diagnostics.",
    )
    ssim_score: Optional[float] = Field(
        default=None,
        description="SSIM in [0, 1] when available; None when not computed.",
    )
    content_change: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Fraction of changed pixels in the content region (status bar "
            "excluded). None when content-diff was not computed."
        ),
    )
    scroll_dx: Optional[float] = Field(
        default=None,
        description="Estimated horizontal translation in pixels; None when not detected.",
    )
    scroll_dy: Optional[float] = Field(
        default=None,
        description="Estimated vertical translation in pixels; None when not detected.",
    )
    signal_counts: ActionEffectSignalCounts = Field(
        default_factory=ActionEffectSignalCounts,
        description=(
            "Diagnostic tally of progress vs no-progress signals that "
            "produced ``status``. Lets observers reproduce the classifier "
            "decision without re-deriving thresholds from raw metrics."
        ),
    )

    @classmethod
    def from_screen_diff(cls, *, diff: Optional[ScreenDiff]) -> "ActionEffect":
        """
        Build an :class:`ActionEffect` from a :class:`ScreenDiff`.

        When no diff is available (capture failure, missing pre-state)
        the effect is reported as ``UNCERTAIN`` with zero raw metrics so downstream consumers don't infer false progress.
        """

        if diff is None:
            return cls(
                phash_distance=0,
                visual_progress=0.0,
                status=ActionEffectStatus.UNCERTAIN,
            )

        visual_progress = cls.__compute_visual_progress(diff=diff)
        scroll_dx = diff.scroll_translation.dx if diff.scroll_translation else None
        scroll_dy = diff.scroll_translation.dy if diff.scroll_translation else None

        status = cls.__classify(diff=diff)
        signal_counts = ActionEffectSignalCounts(
            progress=cls.__count_progress_signals(diff=diff),
            no_progress=cls.__count_no_progress_signals(diff=diff),
            expected=cls.__expected_no_progress_signals(diff=diff),
        )

        return cls(
            status=status,
            scroll_dx=scroll_dx,
            scroll_dy=scroll_dy,
            ssim_score=diff.ssim_score,
            signal_counts=signal_counts,
            visual_progress=visual_progress,
            phash_distance=diff.phash_distance,
            content_change=diff.content_pixel_diff_ratio,
        )

    @staticmethod
    def __compute_visual_progress(*, diff: ScreenDiff) -> float:
        """
        Project the available similarity signals onto a [0, 1] scalar.

        Prefer SSIM-derived progress (``1 - SSIM``) when SSIM is present because it tracks perceptual similarity tightly.
        Fall back to normalized pHash hamming distance otherwise. Either way the result is clamped to [0, 1] so the prompt format stays stable.
        """

        if diff.ssim_score is not None:
            return max(0.0, min(1.0, 1.0 - diff.ssim_score))

        if diff.phash_distance <= 0:
            return 0.0

        normalized = diff.phash_distance / 64.0
        return max(0.0, min(1.0, normalized))

    @staticmethod
    def __classify(*, diff: ScreenDiff) -> ActionEffectStatus:
        """
        Bucket the diff into PROGRESS / NO_PROGRESS / UNCERTAIN.

        PROGRESS fires when any single strong-progress signal is present (large pHash jump, low SSIM, large content diff OR meaningful scroll translation).
        NO_PROGRESS requires every available signal to sit below its no-progress floor. Anything else lands in UNCERTAIN — the agent will treat it as ambiguous
        rather than infer false progress.
        """

        if diff.activity_changed:
            return ActionEffectStatus.PROGRESS

        progress_hits = ActionEffect.__count_progress_signals(diff=diff)
        no_progress_hits = ActionEffect.__count_no_progress_signals(diff=diff)

        if progress_hits >= 1:
            return ActionEffectStatus.PROGRESS

        if no_progress_hits >= ActionEffect.__expected_no_progress_signals(diff=diff):
            return ActionEffectStatus.NO_PROGRESS

        if diff.xml_hash_changed or diff.interaction_hash_changed:
            return ActionEffectStatus.UNCERTAIN

        return ActionEffectStatus.UNCERTAIN

    @staticmethod
    def __count_progress_signals(*, diff: ScreenDiff) -> int:
        """
        Count how many independent metrics indicate forward progress.
        """

        hits = 0

        if diff.phash_distance > ACTION_EFFECT_PROGRESS_PHASH_ABOVE:
            hits += 1

        if diff.ssim_score is not None and diff.ssim_score < ACTION_EFFECT_PROGRESS_SSIM_BELOW:
            hits += 1

        if (
            diff.content_pixel_diff_ratio is not None
            and diff.content_pixel_diff_ratio > ACTION_EFFECT_PROGRESS_CONTENT_DIFF_ABOVE
        ):
            hits += 1

        if diff.scroll_translation is not None:
            dx = abs(diff.scroll_translation.dx)
            dy = abs(diff.scroll_translation.dy)
            if (
                dx > ACTION_EFFECT_PROGRESS_SCROLL_DISTANCE_PX_ABOVE
                or dy > ACTION_EFFECT_PROGRESS_SCROLL_DISTANCE_PX_ABOVE
            ):
                hits += 1

        return hits

    @staticmethod
    def __count_no_progress_signals(*, diff: ScreenDiff) -> int:
        """
        Count how many independent metrics indicate the screen did not meaningfully change.
        """

        hits = 0

        if diff.phash_distance <= ACTION_EFFECT_NO_PROGRESS_PHASH_BELOW_OR_EQ:
            hits += 1

        if diff.ssim_score is not None and diff.ssim_score >= ACTION_EFFECT_NO_PROGRESS_SSIM_ABOVE:
            hits += 1

        if (
            diff.content_pixel_diff_ratio is not None
            and diff.content_pixel_diff_ratio <= ACTION_EFFECT_NO_PROGRESS_CONTENT_DIFF_BELOW_OR_EQ
        ):
            hits += 1

        if diff.scroll_translation is not None:
            dx = abs(diff.scroll_translation.dx)
            dy = abs(diff.scroll_translation.dy)
            if (
                dx <= ACTION_EFFECT_NO_PROGRESS_SCROLL_DISTANCE_PX_BELOW_OR_EQ
                and dy <= ACTION_EFFECT_NO_PROGRESS_SCROLL_DISTANCE_PX_BELOW_OR_EQ
            ):
                hits += 1

        return hits

    @staticmethod
    def __expected_no_progress_signals(*, diff: ScreenDiff) -> int:
        """
        Number of signals that must agree on "no progress" for the classifier to commit to NO_PROGRESS.
        pHash is always present; SSIM, content-diff, and scroll are optional. Requiring all *available*
        signals to agree keeps the classifier conservative — if any metric is unsure we land in UNCERTAIN, not in a false NO_PROGRESS.
        """

        available = 1  # pHash always available

        if diff.ssim_score is not None:
            available += 1

        if diff.content_pixel_diff_ratio is not None:
            available += 1

        if diff.scroll_translation is not None:
            available += 1

        return available


class EffectReading(SealedModel):
    """
    Direction-aware trial facets computed alongside the live effect classification.
    """

    live: ActionEffectStatus = Field(description="Status the live classifier produced this turn.")
    trial: ActionEffectStatus = Field(
        description="Status the direction-aware classifier would have produced."
    )

    scoped: Optional[bool] = Field(
        default=None,
        description=(
            "Whether changed regions covered enough of the action-target region; "
            "None when the diff or target geometry was unavailable."
        ),
    )
    departed: Optional[bool] = Field(
        default=None,
        description="Whether the foreground left the target application; None when unknown.",
    )
    overlap: Optional[float] = Field(
        ge=0.0,
        le=1.0,
        default=None,
        description="Largest fraction of the target region covered by one changed region.",
    )
