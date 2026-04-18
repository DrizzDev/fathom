from __future__ import annotations

from dataclasses import dataclass

DEFAULT_WEB_DRIVER_AGENT_URL = "http://127.0.0.1:8100"


@dataclass(frozen=True)
class IOSGestureDefaults:
    """
    Centralized defaults for iOS gesture behavior.
    """

    back_duration_milliseconds: int = 250
    back_end_x_ratio: float = 0.60
    back_start_x_ratio: float = 0.05
    back_y_ratio: float = 0.50


@dataclass(frozen=True)
class IOSAdapterDefaults:
    """
    Centralized defaults for iOS adapter behavior.
    """

    device_ready_poll_seconds: float = 1.0
    screenshot_minimum_bytes: int = 64
    simulator_control_command: str = "simctl"
    springboard_bundle_identifier: str = "com.apple.springboard"
    unknown_bundle_identifier: str = "unknown"


@dataclass(frozen=True)
class IOSHierarchyDefaults:
    """
    Centralized defaults for iOS hierarchy retrieval.
    """

    source_path: str = "source"
    source_query_path: str = "source?format=xml"
