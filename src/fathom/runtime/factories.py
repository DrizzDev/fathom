from __future__ import annotations

from logging import getLogger
from typing import Callable, Dict, Optional

from fathom.adapters.device.local.adb import ADBDevice
from fathom.adapters.device.local.ios import IOSDevice
from fathom.adapters.device.remote.adb import ADBRemoteDeviceAdapter
from fathom.adapters.device.remote.ios import IOSRemoteDeviceAdapter
from fathom.adapters.interaction.noop import NoopInteraction
from fathom.adapters.interaction.pypika.postgres import PostgresInteraction
from fathom.adapters.interaction.pypika.sqlite import SQLiteInteraction
from fathom.adapters.llm.gemini import GeminiLLM
from fathom.adapters.perception.android import AndroidPerceptionAdapter
from fathom.adapters.perception.ios import (
    IOSEnhancedPerceptionAdapter,
    IOSNativePerceptionAdapter,
)
from fathom.adapters.perception.remote import RemotePerceptionAdapter
from fathom.adapters.scheduler.inprocess import InProcessJobScheduler
from fathom.adapters.scheduler.noop import NoopJobScheduler
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
from fathom.constants.scheduler import JobSchedulerKind
from fathom.constants.storage import InteractionBackend
from fathom.core.exceptions import StorageConfigurationError
from fathom.infrastructure.storage.cloud import GCSImageStorage
from fathom.interfaces.device import DevicePort
from fathom.interfaces.factory import (
    DeviceFactoryPort,
    InteractionFactoryPort,
    JobSchedulerFactoryPort,
    LLMFactoryPort,
    PerceptionFactoryPort,
    SignalFactoryPort,
    TelemetryFactoryPort,
)
from fathom.interfaces.interaction import InteractionPort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.perception import PerceptionPort
from fathom.interfaces.scheduler import JobSchedulerPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import (
    DeviceConfiguration,
    InProcessJobSchedulerConfiguration,
    InteractionStorageConfiguration,
    JobSchedulerConfiguration,
    LLMConfiguration,
    NoopJobSchedulerConfiguration,
    PostgresInteractionConfiguration,
    SQLiteInteractionConfiguration,
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
    ) -> PerceptionPort:
        """
        Create the appropriate PerceptionPort implementation.
        """

        if configuration.type == DeviceConnectionType.REMOTE:
            return RemotePerceptionAdapter(device=device, include_hierarchy=use_xml)

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
            return AndroidPerceptionAdapter(device=device, include_hierarchy=use_xml)

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


class InteractionFactory(InteractionFactoryPort):
    """
    Factory dispatching interaction-storage backends from typed configuration.

    Mirrors DeviceFactory: one backend enum, one nested config per backend,
    factory raises a typed StorageConfigurationError when the matching nested
    configuration is missing. The Pydantic envelope already validates this at
    construction time; the factory checks again so direct callers get the
    same error path as wire-validated requests.
    """

    def create(self, *, configuration: InteractionStorageConfiguration) -> InteractionPort:
        """
        Build the interaction-storage adapter for the selected backend.
        """

        if configuration.backend == InteractionBackend.SQLITE:
            return self.__create_sqlite(configuration=configuration.sqlite)
        if configuration.backend == InteractionBackend.POSTGRES:
            return self.__create_postgres(configuration=configuration.postgres)
        if configuration.backend == InteractionBackend.NOOP:
            return self.__create_noop()

        raise NotImplementedError(
            f"Interaction backend {configuration.backend.value} is not implemented"
        )

    def __create_sqlite(
        self, *, configuration: Optional[SQLiteInteractionConfiguration]
    ) -> InteractionPort:
        """
        Build a SQLite-backed interaction adapter.
        """

        if configuration is None:
            raise StorageConfigurationError(
                backend=InteractionBackend.SQLITE.value,
                message="SQLite interaction storage requires a configuration",
            )
        return SQLiteInteraction(configuration=configuration)

    def __create_postgres(
        self, *, configuration: Optional[PostgresInteractionConfiguration]
    ) -> InteractionPort:
        """
        Build a Postgres-backed interaction adapter.
        """

        if configuration is None:
            raise StorageConfigurationError(
                backend=InteractionBackend.POSTGRES.value,
                message="Postgres interaction storage requires a configuration",
            )
        return PostgresInteraction(configuration=configuration)

    def __create_noop(self) -> InteractionPort:
        """
        Build a noop interaction adapter that swallows writes.
        """

        return NoopInteraction()


class JobSchedulerFactory(JobSchedulerFactoryPort):
    """
    Factory dispatching durable-job schedulers from typed configuration.
    """

    def create(
        self,
        *,
        configuration: JobSchedulerConfiguration,
        interaction: InteractionPort,
    ) -> JobSchedulerPort:
        """
        Build the scheduler adapter for the selected dispatcher kind.
        """

        if configuration.kind == JobSchedulerKind.IN_PROCESS:
            return self.__create_inprocess(
                configuration=configuration.inprocess,
                interaction=interaction,
            )
        if configuration.kind == JobSchedulerKind.NOOP:
            return self.__create_noop(configuration=configuration.noop)

        raise NotImplementedError(f"Job scheduler {configuration.kind.value} is not implemented")

    def __create_inprocess(
        self,
        *,
        configuration: Optional[InProcessJobSchedulerConfiguration],
        interaction: InteractionPort,
    ) -> JobSchedulerPort:
        """
        Build an in-process durable-job scheduler.
        """

        if configuration is None:
            raise StorageConfigurationError(
                backend=JobSchedulerKind.IN_PROCESS.value,
                message="In-process job scheduler requires a configuration",
            )
        return InProcessJobScheduler(
            configuration=configuration,
            interaction=interaction,
        )

    def __create_noop(
        self, *, configuration: Optional[NoopJobSchedulerConfiguration]
    ) -> JobSchedulerPort:
        """
        Build a noop scheduler adapter.
        """

        if configuration is None:
            raise StorageConfigurationError(
                backend=JobSchedulerKind.NOOP.value,
                message="Noop job scheduler requires a configuration",
            )
        return NoopJobScheduler()
