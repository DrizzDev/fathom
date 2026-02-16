"""
Service for saving visual execution traces.
"""

from __future__ import annotations

from datetime import datetime
from logging import getLogger
from typing import TYPE_CHECKING, Any, Tuple

from fathom.processing.annotator import ImageAnnotator

if TYPE_CHECKING:
    from fathom.base.paths import SharedPathManager

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
        image_data: bytes,
        action: Any,
        coords: Tuple[int, ...],
        package_name: str,
        session_id: str,
        step_number: int,
    ) -> None:
        """
        Save an annotated trace image to the session directory.
        
        Args:
            image_data: Raw screenshot bytes.
            action: The executed Action object.
            coords: Coordinates of the action (x, y) or (x1, y1, x2, y2).
            package_name: Target package name for folder organization.
            session_id: Current session identifier.
            step_number: Current step count.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            action_type = getattr(action, "action_type", "unknown")
            if hasattr(action_type, "value"):
                action_type = action_type.value
                
            filename = f"step_{step_number}_{action_type}_{timestamp}.png"
            
            path = self.__path_manager.get_trace_path(
                package_name=package_name,
                session_id=session_id,
                filename=filename,
            )
            
            # ImageAnnotator handles the actual drawing and saving
            ImageAnnotator.trace(
                image_data=image_data,
                output_path=str(path),
                action_type=action_type,
                coords=coords,
                label=getattr(action, "to_description", lambda: "Action")(),
            )
            
        except Exception as exception:
            logger.warning(f"Failed to save trace: {exception}")
