from __future__ import annotations

from enum import StrEnum


class DeviceConnectionType(StrEnum):
    """
    Supported device connection channels.
    """

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


class DevicePlatform(StrEnum):
    """
    Supported execution platforms.
    """

    IOS = "IOS"
    WEB = "WEB"
    ANDROID = "ANDROID"

    REMOTE = "REMOTE"
    DESKTOP = "DESKTOP"


class IOSAutomationBackend(StrEnum):
    """
    Supported iOS automation backends.
    """

    APPIUM = "APPIUM"
    XCUITEST = "XCUITEST"
    XCRUN_SIMCTL = "XCRUN_SIMCTL"
    WEBDRIVER_AGENT = "WEBDRIVER_AGENT"
    IDB = "IDB"
