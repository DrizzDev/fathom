from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class IOSGestureDefaults(BaseModel):
    """
    Centralized defaults for iOS gesture behavior.
    """

    model_config = ConfigDict(frozen=True)

    back_duration_milliseconds: int = 250
    back_end_x_ratio: float = 0.60
    back_start_x_ratio: float = 0.05
    back_y_ratio: float = 0.50


class IOSAdapterDefaults(BaseModel):
    """
    Centralized defaults for iOS adapter behavior.
    """

    model_config = ConfigDict(frozen=True)

    device_ready_poll_seconds: float = 1.0
    screenshot_minimum_bytes: int = 64
    simulator_control_command: str = "simctl"
    springboard_bundle_identifier: str = "com.apple.springboard"
    unknown_bundle_identifier: str = "unknown"


class IOSHierarchyDefaults(BaseModel):
    """
    Centralized defaults for iOS hierarchy retrieval.
    """

    model_config = ConfigDict(frozen=True)

    source_path: str = "source"
    source_query_path: str = "source?format=xml"
