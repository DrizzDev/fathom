from __future__ import annotations

from logging import getLogger
from typing import Callable, Dict

from fathom.adapters.device.adb import ADBDevice
from fathom.adapters.device.ios import IOSDevice
from fathom.adapters.device.remote import RemoteDeviceAdapter
from fathom.adapters.hierarchy.ios import IOSHierarchyAdapterFactory
from fathom.adapters.llm.gemini import GeminiLLM
from fathom.adapters.perception.android import AndroidPerceptionAdapter
from fathom.adapters.perception.ios import (
    IOSEnhancedPerceptionAdapter,
    IOSNativePerceptionAdapter,
)
from fathom.adapters.perception.remote import RemotePerceptionAdapter
from fathom.adapters.signal.interactive import InteractiveSignal
from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.signal.socket import SocketSignal
from fathom.adapters.storage.cloud import CloudStorage
from fathom.adapters.storage.composite import CompositeStorage
from fathom.adapters.storage.local import LocalStorage
from fathom.adapters.telemetry.redis import RedisTelemetryAdapter
from fathom.adapters.telemetry.structlog import StructlogAdapter
from fathom.base.paths import SharedPathManager
from fathom.constants.platform import DeviceConnectionType, DevicePlatform, IOSAutomationBackend
from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.interfaces.device import DevicePort
from fathom.interfaces.factory import (
    DeviceFactoryPort,
    HierarchyFactoryPort,
    LLMFactoryPort,
    PerceptionFactoryPort,
    SignalFactoryPort,
    TelemetryFactoryPort,
)
from fathom.interfaces.hierarchy import HierarchyPort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import (
    DeviceConfiguration,
    IOSConfiguration,
    LLMConfiguration,
    StorageConfiguration,
    TelemetryConfiguration,
)

logger = getLogger(__name__)


class HierarchyFactory(HierarchyFactoryPort):
    """
    Factory for creating hierarchy adapters.
    """

    def __init__(self) -> None:
        self.__ios_factory = IOSHierarchyAdapterFactory()

    def create(self, *, configuration: IOSConfiguration) -> HierarchyPort:
        """
        Create hierarchy adapter from iOS configuration.
        """

        return self.__ios_factory.create(configuration=configuration)


class DeviceFactory(DeviceFactoryPort):
    """
    Factory for creating device adapters based on configuration.
    """

    def create(self, *, configuration: DeviceConfiguration) -> DevicePort:
        """
        Create the appropriate DevicePort implementation.
        """

        if configuration.type == DeviceConnectionType.REMOTE:
            return RemoteDeviceAdapter(configuration=configuration)

        if configuration.platform == DevicePlatform.IOS:
            return IOSDevice(configuration=configuration.ios)

        if configuration.platform == DevicePlatform.ANDROID:
            return ADBDevice(configuration=configuration.android)

        raise NotImplementedError(
            f"Device adapter for platform {configuration.platform} is not implemented"
        )


class LLMFactory(LLMFactoryPort):
    """
    Factory for creating LLM adapters.
    """

    def create(self, *, configuration: LLMConfiguration) -> LLMPort:
        """
        Create LLM adapter from runtime LLM configuration.
        """

        if configuration.provider == "gemini":
            return GeminiLLM(configuration=configuration)

        raise NotImplementedError(
            f"LLM adapter for provider {configuration.provider} is not implemented"
        )


class PerceptionFactory(PerceptionFactoryPort):
    """
    Factory for creating perception adapters based on configuration and device binding.
    """

    def __init__(self, *, hierarchy_factory: HierarchyFactoryPort | None = None) -> None:
        self.__hierarchy_factory = hierarchy_factory or HierarchyFactory()

    def create(self, *, configuration: DeviceConfiguration, device: DevicePort) -> PerceptionPort:
        """
        Create the appropriate PerceptionPort implementation.
        """

        if configuration.type == DeviceConnectionType.REMOTE:
            if not isinstance(device, RemoteDeviceAdapter):
                raise TypeError("Remote perception requires RemoteDeviceAdapter.")
            return RemotePerceptionAdapter(device=device)

        if configuration.platform == DevicePlatform.IOS:
            if not isinstance(device, IOSDevice):
                raise TypeError("iOS perception requires IOSDevice.")
            backend = configuration.ios.automation_backend
            if backend == IOSAutomationBackend.XCRUN_SIMCTL:
                return IOSNativePerceptionAdapter(device=device)
            if backend in {
                IOSAutomationBackend.XCUITEST,
                IOSAutomationBackend.WEBDRIVER_AGENT,
            }:
                hierarchy_adapter = self.__hierarchy_factory.create(configuration=configuration.ios)
                return IOSEnhancedPerceptionAdapter(
                    device=device,
                    hierarchy=hierarchy_adapter,
                )
            raise NotImplementedError(
                f"Local iOS perception strategy is not implemented for backend {backend.value}"
            )

        if configuration.platform == DevicePlatform.ANDROID:
            if not isinstance(device, ADBDevice):
                raise TypeError("Android perception requires ADBDevice.")
            return AndroidPerceptionAdapter(device=device)

        raise NotImplementedError(
            f"Perception adapter for platform {configuration.platform} is not implemented"
        )


class SignalFactory(SignalFactoryPort):
    """
    Factory for creating signal adapters.
    """

    __SOCKET_MODE = "socket"

    def create(self, *, interactive: bool, signal_type: str) -> SignalPort:
        """
        Create signal adapter from interaction mode and signal type.
        """

        if not interactive:
            return NoopSignal()

        if signal_type == self.__SOCKET_MODE:
            return SocketSignal()

        return InteractiveSignal()


class TelemetryFactory(TelemetryFactoryPort):
    """
    Factory for creating telemetry adapters based on configuration.
    """

    def create(self, *, configuration: TelemetryConfiguration) -> TelemetryPort:
        """
        Create the appropriate TelemetryPort implementation.
        """

        if configuration.type == "REDIS":
            return RedisTelemetryAdapter(configuration=configuration)

        return StructlogAdapter()


class StorageFactory:
    """
    Factory for creating storage adapters based on configuration.
    """

    def create(
        self, *, configuration: StorageConfiguration, path_manager: SharedPathManager
    ) -> StoragePort:
        """
        Create the appropriate StoragePort implementation.
        Supports LOCAL, CLOUD, or both using CompositeStorage.
        """

        strategy_map: Dict[str, Callable[[], StoragePort]] = {
            "LOCAL": lambda: LocalStorage(path_manager=path_manager),
            "CLOUD": lambda: CloudStorage(storage=GCSImageStorage(configuration=configuration)),
        }

        active_storages = []

        for backend in configuration.backends:
            if backend == "CLOUD" and not configuration.storage_bucket:
                logger.warning("CLOUD backend specified without a storage_bucket, skipping.")
                continue

            if creator := strategy_map.get(backend):
                active_storages.append(creator())

        if not active_storages:
            logger.warning("No valid storage backends configured, defaulting to LOCAL.")
            return LocalStorage(path_manager=path_manager)

        if len(active_storages) == 1:
            return active_storages[0]

        return CompositeStorage(storages=active_storages)
