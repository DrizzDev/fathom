from __future__ import annotations

from logging import getLogger
from typing import Optional, Tuple

from fathom.constants.screen import ACTION_REGION_CHANGE_FLOOR
from fathom.schemas.actions import Bounds
from fathom.schemas.effect import ActionEffect, ActionEffectStatus, EffectReading
from fathom.schemas.screens import ScreenChangeRegion, ScreenDiff

logger = getLogger(__name__)


class EffectClassifier:
    """
    Derives direction-aware trial facets for one action effect without altering the live classification.
    """

    def classify(
        self,
        *,
        effect: ActionEffect,
        diff: Optional[ScreenDiff],
        bounds: Optional[Bounds],
        package: str,
        foreground: str,
    ) -> EffectReading:
        """
        Return the trial reading derived from region scope and foreground direction.
        """

        scoped, overlap = self.__scoped(diff=diff, bounds=bounds)
        departed = self.__departed(package=package, foreground=foreground)

        return EffectReading(
            live=effect.status,
            trial=self.__trial(live=effect.status, scoped=scoped, departed=departed),
            scoped=scoped,
            departed=departed,
            overlap=overlap,
        )

    @staticmethod
    def __trial(
        *,
        live: ActionEffectStatus,
        scoped: Optional[bool],
        departed: Optional[bool],
    ) -> ActionEffectStatus:
        """
        Apply direction first, then scoped promotion, then fall back to the live status.
        """

        if departed is True:
            return ActionEffectStatus.REGRESSION

        if scoped is True and live is not ActionEffectStatus.PROGRESS:
            return ActionEffectStatus.PROGRESS

        return live

    @classmethod
    def __scoped(
        cls,
        *,
        diff: Optional[ScreenDiff],
        bounds: Optional[Bounds],
    ) -> Tuple[Optional[bool], Optional[float]]:
        """
        Return whether change concentrated on the target region, with the covering fraction.
        """

        if diff is None or bounds is None or bounds.width <= 0 or bounds.height <= 0:
            return None, None

        overlap = max(
            (cls.__coverage(region=region, bounds=bounds) for region in diff.changed_regions),
            default=0.0,
        )

        return overlap >= ACTION_REGION_CHANGE_FLOOR, overlap

    @staticmethod
    def __coverage(*, region: ScreenChangeRegion, bounds: Bounds) -> float:
        """
        Return the fraction of the target bounds covered by one changed region.
        """

        width = min(region.x + region.width, bounds.x + bounds.width) - max(region.x, bounds.x)
        height = min(region.y + region.height, bounds.y + bounds.height) - max(region.y, bounds.y)

        if width <= 0 or height <= 0:
            return 0.0

        return (width * height) / (bounds.width * bounds.height)

    @staticmethod
    def __departed(*, package: str, foreground: str) -> Optional[bool]:
        """
        Return whether the foreground left the target application, None when unknown.
        """

        if not package or not foreground:
            return None

        return foreground != package


class EffectRecorder:
    """
    Computes and records the trial effect reading without touching the live classification.
    """

    def __init__(self, *, classifier: Optional[EffectClassifier] = None) -> None:
        """
        Bind the trial classifier whose readings are recorded.
        """

        self.__classifier = classifier if classifier is not None else EffectClassifier()

    def observe(
        self,
        *,
        workflow_id: str,
        effect: ActionEffect,
        diff: Optional[ScreenDiff],
        bounds: Optional[Bounds],
        package: str,
        foreground: str,
    ) -> Optional[EffectReading]:
        """
        Classify the trial reading and log it; never raise into the live loop.
        """

        try:
            reading = self.__classifier.classify(
                effect=effect,
                diff=diff,
                bounds=bounds,
                package=package,
                foreground=foreground,
            )
        except Exception as exception:
            logger.warning(
                "Effect trial classification failed; live effect unaffected",
                extra={
                    "event": "effect.trial.failed",
                    "workflow.id": workflow_id,
                    "exception.type": type(exception).__name__,
                    "exception.message": str(exception),
                },
            )
            return None

        logger.info(
            "Effect trial compared",
            extra={
                "event": "effect.trial.compared",
                "workflow.id": workflow_id,
                "effect.agrees": reading.live is reading.trial,
                "effect.reading": reading.model_dump(mode="json"),
            },
        )
        return reading
