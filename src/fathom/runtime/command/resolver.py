from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from fathom.constants.ios import DEFAULT_WEB_DRIVER_AGENT_URL
from fathom.constants.platform import DeviceConnectionType, DevicePlatform, IOSAutomationBackend
from fathom.schemas.cli import LocalCommandInput
from fathom.schemas.configuration import (
    ADBConfiguration,
    DeviceConfiguration,
    IOSConfiguration,
    RemoteDeviceConfiguration,
)
from fathom.settings.env import FathomSettings


class LocalDeviceConfigurationResolverPort(ABC):
    """
    Port for resolving local device configuration from command input.
    """

    @abstractmethod
    def resolve(
        self,
        *,
        command_input: LocalCommandInput,
        settings: FathomSettings,
    ) -> DeviceConfiguration:
        """
        Resolve local device configuration for command execution.
        """

        raise NotImplementedError


class RuntimeDeviceDefaultsResolverPort(ABC):
    """
    Port for applying runtime defaults into device configuration.
    """

    @abstractmethod
    def resolve(self, *, configuration: DeviceConfiguration) -> DeviceConfiguration:
        """
        Apply runtime defaults without mutating input configuration.
        """

        raise NotImplementedError


class DeviceConfigurationStrategy(ABC):
    """
    Strategy for platform-specific local device configuration creation.
    """

    @abstractmethod
    def supports(self, *, platform: DevicePlatform) -> bool:
        """
        Check whether strategy supports the provided platform.
        """

        raise NotImplementedError

    @abstractmethod
    def create(
        self,
        *,
        command_input: LocalCommandInput,
        settings: FathomSettings,
    ) -> DeviceConfiguration:
        """
        Create platform-specific device configuration.
        """

        raise NotImplementedError


class IOSDeviceConfigurationStrategy(DeviceConfigurationStrategy):
    """
    iOS local device configuration strategy.
    """

    def supports(self, *, platform: DevicePlatform) -> bool:
        """
        Return whether iOS platform is supported by this strategy.
        """

        return platform == DevicePlatform.IOS

    def create(
        self,
        *,
        command_input: LocalCommandInput,
        settings: FathomSettings,
    ) -> DeviceConfiguration:
        """
        Create iOS local device configuration.
        """

        return DeviceConfiguration(
            platform=DevicePlatform.IOS,
            type=DeviceConnectionType.LOCAL,
            ios=IOSConfiguration(
                executable_path=command_input.ios_executable_path or "xcrun",
                device_identifier=(
                    command_input.ios_device_identifier or command_input.serial_number
                ),
                bundle_identifier=command_input.ios_bundle_identifier,
                automation_backend=(
                    command_input.ios_automation_backend or IOSAutomationBackend.XCRUN_SIMCTL
                ),
                web_driver_agent_url=(
                    command_input.ios_web_driver_agent_url or DEFAULT_WEB_DRIVER_AGENT_URL
                ),
                web_driver_agent_bundle_identifier=command_input.ios_web_driver_agent_bundle_identifier,
                web_driver_agent_request_timeout_seconds=(
                    command_input.ios_web_driver_agent_request_timeout_seconds or 15.0
                ),
            ),
            remote=RemoteDeviceConfiguration(),
        )


class AndroidDeviceConfigurationStrategy(DeviceConfigurationStrategy):
    """
    Android local device configuration strategy.
    """

    def supports(self, *, platform: DevicePlatform) -> bool:
        """
        Return whether Android platform is supported by this strategy.
        """

        return platform == DevicePlatform.ANDROID

    def create(
        self,
        *,
        command_input: LocalCommandInput,
        settings: FathomSettings,
    ) -> DeviceConfiguration:
        """
        Create Android local device configuration.
        """

        return DeviceConfiguration(
            type=DeviceConnectionType.LOCAL,
            platform=DevicePlatform.ANDROID,
            android=ADBConfiguration(
                serial_number=command_input.serial_number or settings.android_serial,
                executable_path=command_input.adb_executable_path or settings.adb_path,
            ),
            remote=RemoteDeviceConfiguration(),
        )


class LocalDeviceConfigurationResolver(LocalDeviceConfigurationResolverPort):
    """
    Resolver that delegates local device configuration creation to platform strategies.
    """

    def __init__(self) -> None:
        self.__strategies: List[DeviceConfigurationStrategy] = [
            IOSDeviceConfigurationStrategy(),
            AndroidDeviceConfigurationStrategy(),
        ]

    def resolve(
        self,
        *,
        command_input: LocalCommandInput,
        settings: FathomSettings,
    ) -> DeviceConfiguration:
        """
        Resolve local device configuration through the matching strategy.
        """

        for strategy in self.__strategies:
            if strategy.supports(platform=command_input.platform):
                return strategy.create(
                    command_input=command_input,
                    settings=settings,
                )

        raise NotImplementedError(
            f"Unsupported local device platform: {command_input.platform.value}"
        )


class RuntimeDeviceDefaultsStrategy(ABC):
    """
    Strategy for runtime default injection into device configuration.
    """

    @abstractmethod
    def supports(self, *, platform: DevicePlatform) -> bool:
        """
        Check whether strategy supports the provided platform.
        """

        raise NotImplementedError

    @abstractmethod
    def apply(self, *, configuration: DeviceConfiguration) -> DeviceConfiguration:
        """
        Apply defaults and return resolved device configuration.
        """

        raise NotImplementedError


class IOSRuntimeDeviceDefaultsStrategy(RuntimeDeviceDefaultsStrategy):
    """
    Runtime defaults strategy for iOS local device configuration.
    """

    def supports(self, *, platform: DevicePlatform) -> bool:
        """
        Return whether iOS platform is supported by this strategy.
        """

        return platform == DevicePlatform.IOS

    def apply(self, *, configuration: DeviceConfiguration) -> DeviceConfiguration:
        """
        Apply iOS runtime defaults.
        """

        resolved = configuration.model_copy(deep=True)
        if resolved.ios.executable_path is None:
            resolved.ios.executable_path = "xcrun"
        return resolved


class AndroidRuntimeDeviceDefaultsStrategy(RuntimeDeviceDefaultsStrategy):
    """
    Runtime defaults strategy for Android local device configuration.
    """

    def __init__(self, *, settings: FathomSettings) -> None:
        self.__settings = settings

    def supports(self, *, platform: DevicePlatform) -> bool:
        """
        Return whether Android platform is supported by this strategy.
        """

        return platform == DevicePlatform.ANDROID

    def apply(self, *, configuration: DeviceConfiguration) -> DeviceConfiguration:
        """
        Apply Android runtime defaults.
        """

        resolved = configuration.model_copy(deep=True)
        if resolved.android.executable_path is None:
            resolved.android.executable_path = self.__settings.adb_path
        return resolved


class RuntimeDeviceDefaultsResolver(RuntimeDeviceDefaultsResolverPort):
    """
    Resolver for runtime default injection via platform strategies.
    """

    def __init__(self, *, settings: FathomSettings) -> None:
        self.__settings = settings
        self.__strategies: Dict[DevicePlatform, RuntimeDeviceDefaultsStrategy] = {
            DevicePlatform.IOS: IOSRuntimeDeviceDefaultsStrategy(),
            DevicePlatform.ANDROID: AndroidRuntimeDeviceDefaultsStrategy(settings=settings),
        }

    def resolve(self, *, configuration: DeviceConfiguration) -> DeviceConfiguration:
        """
        Resolve runtime defaults for local device configuration.
        """

        if configuration.type != DeviceConnectionType.LOCAL:
            return configuration.model_copy(deep=True)

        strategy = self.__strategies.get(configuration.platform)
        if strategy is None:
            return configuration.model_copy(deep=True)

        return strategy.apply(configuration=configuration)
