from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RemoteInteractionRequest(BaseModel):
    """
    Standardized payload for executing actions on a remote device provider.
    Adapters should map internal Action objects to this schema.
    """

    action: str = Field(description="Action type (tap, type, swipe, back, home)")
    execution_id: Optional[str] = Field(default=None, description="The unique ID for the current execution/workflow.")

    x: Optional[int] = Field(default=None, description="Center X coordinate")
    y: Optional[int] = Field(default=None, description="Center Y coordinate")
    points: Optional[List[int]] = Field(
        default=None, description="Swipe trajectory points [x1, y1, x2, y2]"
    )

    text: Optional[str] = Field(default=None, description="Input text for typing actions")
    extras: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific metadata")
