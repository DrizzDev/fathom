from __future__ import annotations

from enum import IntEnum, StrEnum


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


class AndroidKeycode(IntEnum):
    """
    Android SDK keycodes used by the local ADB adapter.
    """

    A = 29
    DEL = 67
    MOVE_END = 123
    DPAD_RIGHT = 22
    CTRL_LEFT = 113


class AndroidClearStrategy(IntEnum):
    """
    Thresholds for the local ADB text-clear strategy.
    """

    DELETE_COUNT = 150
    MODERN_MIN_SDK = 30
    RIGHT_ARROW_COUNT = 20


class IOSClearStrategy(IntEnum):
    """
    Thresholds for WDA-based text clearing.
    """

    MAX_LENGTH = 10_000


class IOSAutomationBackend(StrEnum):
    """
    Supported iOS automation backends.
    """

    APPIUM = "APPIUM"
    XCUITEST = "XCUITEST"
    XCRUN_SIMCTL = "XCRUN_SIMCTL"
    WEBDRIVER_AGENT = "WEBDRIVER_AGENT"
