from __future__ import annotations

from datetime import datetime, timezone
from logging import getLogger
from typing import Any, Tuple

from fathom.base.paths import SharedPathManager
from fathom.constants.artifact import ArtifactFilename
from fathom.processing.annotator import ImageAnnotator
from fathom.schemas.artifact import ArtifactKind

logger = getLogger(__name__)


class TraceService:
    """
    Service for generating and persisting annotated action traces.

    .. deprecated::
        Trace persistence moved to
        :class:`fathom.core.artifact.pipeline.ArtifactPipeline` via
        :class:`fathom.schemas.artifact.TracePayload`. The action
        executor emits trace records directly through the pipeline.
        This service is kept for ad-hoc tooling; do not introduce new
        callers.
    """

    def __init__(self, path_manager: SharedPathManager) -> None:
        self.__path_manager = path_manager

    def save(
        self,
        *,
        action: Any,
        session_id: str,
        step_number: int,
        package_name: str,
        image_data: bytes,
        coords: Tuple[int, ...],
    ) -> None:
        """
        Save an annotated trace image to the session directory.

        Filenames follow the canonical artifact grammar
        ``step-NNN__<kind>__<iso-timestamp-utc>.<ext>`` so even ad-hoc
        traces written through this deprecated path interleave
        correctly in directory listings with pipeline-written
        artifacts.

        .. deprecated::
            Emit a :class:`fathom.schemas.artifact.TracePayload` via
            :class:`ArtifactPipeline` instead.
        """

        import warnings

        warnings.warn(
            "TraceService.save is deprecated; emit a TracePayload via ArtifactPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            action_type = getattr(action, "action_type", "unknown")

            separator = ArtifactFilename.SEPARATOR
            step = str(step_number).zfill(ArtifactFilename.STEP_DIGITS)
            timestamp = datetime.now(tz=timezone.utc).strftime(ArtifactFilename.TIMESTAMP_FORMAT)

            filename = (
                f"step-{step}{separator}{ArtifactKind.TRACE.value}"
                f"{separator}{package_name}{separator}{timestamp}.png"
            )

            path = self.__path_manager.get_trace_path(
                filename=filename,
                session_id=session_id,
            )

            # ImageAnnotator handles the actual drawing and saving
            ImageAnnotator.trace(
                coords=coords,
                output_path=str(path),
                image_data=image_data,
                action_type=action_type,
                label=getattr(action, "to_description", lambda: "Action")(),
            )

        except Exception as exception:
            logger.warning(f"Failed to save trace: {exception}")
