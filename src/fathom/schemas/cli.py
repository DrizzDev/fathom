from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fathom.constants.platform import DevicePlatform, IOSAutomationBackend


class LocalCommandInput(BaseModel):
    """
    Validated local-device command payload parsed from CLI arguments.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    platform: DevicePlatform = Field(default=DevicePlatform.ANDROID)
    serial_number: Optional[str] = Field(default=None, alias="serial")
    adb_executable_path: Optional[str] = Field(default=None, alias="adb_path")
    ios_device_identifier: Optional[str] = Field(default=None)
    ios_bundle_identifier: Optional[str] = Field(default=None)
    ios_executable_path: Optional[str] = Field(default=None)
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
    max_steps: int = Field(default=100, ge=1)
    interactive: bool = Field(default=False)
    realignment_budget: int = Field(default=3, ge=0)
    immediate_realignment: bool = Field(default=True)
    api_key: Optional[str] = Field(default=None)
    verbose: bool = Field(default=False)

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
    max_steps: int = Field(default=100, ge=1)
    verbose: bool = Field(default=False)
