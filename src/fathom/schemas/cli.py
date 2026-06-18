from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fathom.constants.platform import APP_IDENTIFIER_PATTERN, DevicePlatform, IOSAutomationBackend


class LocalCommandInput(BaseModel):
    """
    Validated local-device command payload parsed from CLI arguments.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    platform: DevicePlatform = Field(default=DevicePlatform.ANDROID)
    serial_number: Optional[str] = Field(default=None, alias="serial")
    adb_executable_path: Optional[str] = Field(default=None, alias="adb_path")

    ios_executable_path: Optional[str] = Field(default=None)
    ios_device_identifier: Optional[str] = Field(default=None)
    ios_bundle_identifier: Optional[str] = Field(default=None)

    ios_automation_backend: Optional[IOSAutomationBackend] = Field(default=None)
    ios_web_driver_agent_url: Optional[str] = Field(default=None)
    ios_web_driver_agent_bundle_identifier: Optional[str] = Field(default=None)
    ios_web_driver_agent_request_timeout_seconds: Optional[float] = Field(default=None, gt=0.0)

    @field_validator("platform", mode="before")
    @classmethod
    def __validate_platform(cls, value: object) -> DevicePlatform:
        """
        Validate CLI platform input for local device execution.
        """

        if value is None:
            return DevicePlatform.ANDROID

        if isinstance(value, DevicePlatform):
            platform = value
        else:
            normalized = str(value).strip().upper()
            platform = DevicePlatform(normalized)

        if platform in {DevicePlatform.ANDROID, DevicePlatform.IOS}:
            return platform

        raise ValueError("CLI platform must be Android or iOS for local execution.")

    @field_validator(
        "serial_number",
        "adb_executable_path",
        "ios_device_identifier",
        "ios_bundle_identifier",
        "ios_executable_path",
        "ios_web_driver_agent_url",
        "ios_web_driver_agent_bundle_identifier",
        mode="before",
    )
    @classmethod
    def __normalize_optional_text(cls, value: object) -> Optional[str]:
        """
        Normalize optional CLI text values into trimmed strings.
        """

        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None

    @field_validator("ios_automation_backend", mode="before")
    @classmethod
    def __validate_ios_automation_backend(cls, value: object) -> Optional[IOSAutomationBackend]:
        """
        Validate optional iOS automation backend selection.
        """

        if value is None:
            return None

        if isinstance(value, IOSAutomationBackend):
            return value

        normalized = str(value).strip().upper()
        return IOSAutomationBackend(normalized)


class RunCommandInput(LocalCommandInput):
    """
    Validated command payload for `fathom run`.
    """

    command: Literal["run"] = Field(default="run")

    intent: str = Field(..., min_length=1)
    use_xml: bool = Field(default=False)

    signal: Literal["interactive", "socket"] = Field(default="interactive")

    max_steps: int = Field(default=50, ge=1)
    interactive: bool = Field(default=False)

    realignment_budget: int = Field(default=3, ge=0)
    immediate_realignment: bool = Field(default=True)

    api_key: Optional[str] = Field(default=None)

    verbose: bool = Field(default=False)
    log_file: Optional[str] = Field(
        default=None,
        description="When set, also mirror logs to a file under logs/<date>/<workflow_id>/run.log "
        "(use 'auto' for the default path, or provide an explicit path).",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def __normalize_api_key(cls, value: object) -> Optional[str]:
        """
        Normalize API key input from CLI.
        """

        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None


class ExploreCommandInput(LocalCommandInput):
    """
    Validated command payload for `fathom explore`.
    """

    command: Literal["explore"] = Field(default="explore")
    max_steps: int = Field(default=50, ge=1)
    verbose: bool = Field(default=False)
    tui: bool = Field(default=False, description="Render the run in the live exploration TUI")
    package_name: Optional[str] = Field(
        default=None, description="Application identifier to launch and explore"
    )
    focus: Optional[str] = Field(
        default=None, description="Steer exploration toward a specific flow or feature"
    )

    @field_validator("package_name", mode="before")
    @classmethod
    def __validate_package_name(cls, value: object) -> Optional[str]:
        """
        Normalize and validate the optional target application identifier.
        """

        if value is None:
            return None

        normalized = str(value).strip()
        if not normalized:
            return None

        if not re.fullmatch(APP_IDENTIFIER_PATTERN, normalized):
            raise ValueError(f"Invalid package identifier: {normalized!r}")

        return normalized

    @field_validator("focus", mode="before")
    @classmethod
    def __normalize_focus(cls, value: object) -> Optional[str]:
        """
        Normalize the optional exploration focus into a trimmed string or None.
        """

        if value is None:
            return None

        normalized = str(value).strip()
        return normalized or None
