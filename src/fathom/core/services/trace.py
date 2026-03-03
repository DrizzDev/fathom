from __future__ import annotations

from datetime import datetime
from logging import getLogger
from typing import Any, Tuple

from fathom.base.paths import SharedPathManager
from fathom.processing.annotator import ImageAnnotator

logger = getLogger(__name__)


class TraceService:
    """
    Service for generating and persisting annotated action traces.
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
        """

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            action_type = getattr(action, "action_type", "unknown")

            if hasattr(action_type, "value"):
                action_type = action_type.value

            filename = f"step_{step_number}_{action_type}_{timestamp}.png"

            path = self.__path_manager.get_trace_path(
                filename=filename,
                session_id=session_id,
                package_name=package_name,
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
