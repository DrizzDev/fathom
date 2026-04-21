from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IOSGestureDefaults(BaseModel):
    """
    Centralized defaults for iOS gesture behavior.
    """

    model_config = ConfigDict(frozen=True)

    back_duration: int = Field(default=250, description="Back gesture duration in milliseconds.")

    back_end_x_ratio: float = Field(
        default=0.60,
        description="Horizontal end position for the back gesture as a screen-width ratio.",
    )
    back_start_x_ratio: float = Field(
        default=0.05,
        description="Horizontal start position for the back gesture as a screen-width ratio.",
    )
    back_y_ratio: float = Field(
        default=0.50,
        description="Vertical position for the back gesture as a screen-height ratio.",
    )


class IOSAdapterDefaults(BaseModel):
    """
    Centralized defaults for iOS adapter behavior.
    """

    model_config = ConfigDict(frozen=True)

    device_ready_poll_seconds: float = Field(
        default=1.0,
        description="Delay between simulator readiness checks in seconds.",
    )
    screenshot_minimum_bytes: int = Field(
        default=64,
        description="Minimum screenshot payload size considered valid in bytes.",
    )
    simulator_control_command: str = Field(
        default="simctl",
        description="Simulator control subcommand used through xcrun.",
    )
    springboard_bundle_identifier: str = Field(
        default="com.apple.springboard",
        description="Bundle identifier used to navigate to the iOS home screen.",
    )
    unknown_bundle_identifier: str = Field(
        default="unknown",
        description="Fallback bundle identifier when the active app cannot be resolved.",
    )


class IOSHierarchyDefaults(BaseModel):
    """
    Centralized defaults for iOS hierarchy retrieval.
    """

    model_config = ConfigDict(frozen=True)

    source_path: str = Field(
        default="source",
        description="WebDriverAgent endpoint path for hierarchy source retrieval.",
    )
    source_query_path: str = Field(
        default="source?format=xml",
        description="WebDriverAgent endpoint path for XML hierarchy source retrieval.",
    )
