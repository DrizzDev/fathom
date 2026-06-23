from __future__ import annotations

from logging import getLogger
from typing import Callable, Dict, Optional

from fathom.adapters.device.local.adb import ADBDevice
from fathom.adapters.device.local.ios import IOSDevice
from fathom.adapters.device.remote.adb import ADBRemoteDeviceAdapter
from fathom.adapters.device.remote.ios import IOSRemoteDeviceAdapter
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
from fathom.adapters.signal.temporal import TemporalSignalAdapter
from fathom.adapters.storage.cloud import CloudStorage
from fathom.adapters.storage.composite import CompositeStorage
from fathom.adapters.storage.local import LocalStorage
from fathom.adapters.telemetry.redis import RedisTelemetryAdapter
from fathom.adapters.telemetry.structlog import StructlogAdapter
from fathom.base.paths import SharedPathManager
from fathom.constants.platform import DeviceConnectionType, DevicePlatform, IOSAutomationBackend
from fathom.constants.run import SignalAdapterType
from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.interfaces.device import DevicePort
from fathom.interfaces.factory import (
    DeviceFactoryPort,
    LLMFactoryPort,
    PerceptionFactoryPort,
    SignalFactoryPort,
    TelemetryFactoryPort,
)
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import (
    DeviceConfiguration,
    LLMConfiguration,
    StorageConfiguration,
    TelemetryConfiguration,
)

logger = getLogger(__name__)


class DeviceFactory(DeviceFactoryPort):
    """
    Factory for creating device adapters based on configuration.
    """

    def create(self, *, configuration: DeviceConfiguration) -> DevicePort:
        """
        Create the appropriate DevicePort implementation.
        """

        if configuration.type == DeviceConnectionType.REMOTE:
            if configuration.platform == DevicePlatform.IOS:
                return IOSRemoteDeviceAdapter(configuration=configuration)

            if configuration.platform == DevicePlatform.ANDROID:
                return ADBRemoteDeviceAdapter(configuration=configuration)

            raise NotImplementedError(
                f"Remote device adapter for platform {configuration.platform} is not implemented"
            )

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

    def create(
        self,
        *,
        configuration: DeviceConfiguration,
        device: DevicePort,
        use_xml: bool,
        capture_hierarchy: bool = False,
    ) -> PerceptionPort:
        """
        Create the appropriate PerceptionPort implementation.

        ``use_xml`` selects XML-grounded perception for the agent; the separate
        ``capture_hierarchy`` only adds the view hierarchy to each capture (for
        structural screen dedup) without changing how the agent grounds.
        """

        include_hierarchy = use_xml or capture_hierarchy

        if configuration.type == DeviceConnectionType.REMOTE:
            return RemotePerceptionAdapter(device=device, include_hierarchy=include_hierarchy)

        if configuration.platform == DevicePlatform.IOS:
            backend = configuration.ios.automation_backend
            if backend == IOSAutomationBackend.XCRUN_SIMCTL:
                if use_xml:
                    return IOSEnhancedPerceptionAdapter(device=device)
                return IOSNativePerceptionAdapter(device=device)
            if not use_xml:
                return IOSNativePerceptionAdapter(device=device)
            if backend in {
                IOSAutomationBackend.XCUITEST,
                IOSAutomationBackend.WEBDRIVER_AGENT,
            }:
                return IOSEnhancedPerceptionAdapter(device=device)
            raise NotImplementedError(
                f"Local iOS perception strategy is not implemented for backend {backend.value}"
            )

        if configuration.platform == DevicePlatform.ANDROID:
            return AndroidPerceptionAdapter(device=device, include_hierarchy=include_hierarchy)

        raise NotImplementedError(
            f"Perception adapter for platform {configuration.platform} is not implemented"
        )


class SignalFactory(SignalFactoryPort):
    """
    Factory for creating signal adapters.
    """

    def create(
        self,
        *,
        signal_type: str,
        interactive: bool,
        workflow_id: Optional[str] = None,
    ) -> SignalPort:
        """
        Create signal adapter from interaction mode and signal type.

        When workflow_id is provided, the adapter runs inside a Temporal activity
        and always resolves to TemporalSignalAdapter regardless of signal_type.
        """

        if not interactive:
            return NoopSignal()

        if workflow_id is not None:
            return TemporalSignalAdapter(workflow_id=workflow_id)

        if signal_type == SignalAdapterType.SOCKET:
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
