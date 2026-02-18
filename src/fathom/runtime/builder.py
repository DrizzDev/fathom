from __future__ import annotations

from typing import Optional

from fathom.adapters.knowledge.sqlite import SQLiteKnowledge
from fathom.adapters.memory.sqlite import SQLiteMemory
from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.storage.local import LocalStorage
from fathom.adapters.telemetry.structlog import StructlogAdapter
from fathom.base.paths import SharedPathManager
from fathom.core.exceptions import ConfigurationError
from fathom.interfaces.device import DevicePort
from fathom.interfaces.knowledge import KnowledgePort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.signal import SignalPort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.runtime.runner import FathomRunner
from fathom.schemas.configuration import (
    ExecutionConfiguration,
    ExplorationConfiguration,
    FathomConfiguration,
    IntentConfiguration,
)
from fathom.settings.env import FathomSettings


class FathomBuilder:
    """
    Fluent builder for Fathom configuration.
    Methods are order-independent. Validation happens at build().
    """

    def __init__(self, path_manager: Optional[SharedPathManager] = None) -> None:
        """
        Initialize builder with no ports configured.
        """

        self.__device: Optional[DevicePort] = None

        self.__llm: Optional[LLMPort] = None
        self.__memory: Optional[MemoryPort] = None
        self.__knowledge: Optional[KnowledgePort] = None

        self.__signal: Optional[SignalPort] = None
        self.__storage: Optional[StoragePort] = None
        self.__telemetry: Optional[TelemetryPort] = None

        self.__config: FathomConfiguration = FathomConfiguration()
        self.__path_manager = path_manager or SharedPathManager(settings=FathomSettings())

    def device(self, device: DevicePort) -> FathomBuilder:
        """
        Configure device port (plug-and-play).
        """

        self.__device = device
        return self

    def llm(self, llm: LLMPort) -> FathomBuilder:
        """
        Configure LLM port (plug-and-play).
        """

        self.__llm = llm
        return self

    def memory(self, memory: MemoryPort) -> FathomBuilder:
        """
        Configure memory port (plug-and-play).
        """

        self.__memory = memory
        return self

    def knowledge(self, knowledge: KnowledgePort) -> FathomBuilder:
        """
        Configure knowledge port (plug-and-play).
        """

        self.__knowledge = knowledge
        return self

    def signal(self, signal: SignalPort) -> FathomBuilder:
        """
        Configure signal port (plug-and-play).
        """

        self.__signal = signal
        return self

    def storage(self, storage: StoragePort) -> FathomBuilder:
        """
        Configure storage port (plug-and-play).
        """

        self.__storage = storage
        return self

    def telemetry(self, telemetry: TelemetryPort) -> FathomBuilder:
        """
        Configure telemetry port (plug-and-play).
        """

        self.__telemetry = telemetry
        return self

    def config(self, config: FathomConfiguration) -> FathomBuilder:
        """
        Configure Fathom settings.
        """

        self.__config = config
        return self

    def execution_config(self, config: ExecutionConfiguration) -> FathomBuilder:
        """
        Configure execution engine settings.
        """

        self.__config.engine = config
        return self

    def intent_config(self, config: IntentConfiguration) -> FathomBuilder:
        """
        Configure intent strategy settings.
        """

        self.__config.intent = config
        return self

    def exploration_config(self, config: ExplorationConfiguration) -> FathomBuilder:
        """
        Configure exploration strategy settings.
        """

        self.__config.exploration = config
        return self

    def build(self) -> FathomRunner:
        """
        Build configured Fathom instance.

        Validates required ports and applies defaults.

        Returns:
            FathomRunner instance with all ports configured

        Raises:
            ConfigurationError: If required ports (device, llm) are not configured
        """

        # Ensure path manager exists
        if not self.__path_manager:
            self.__path_manager = SharedPathManager(settings=FathomSettings())

        # Validate required ports
        if not self.__device:
            raise ConfigurationError("Device port is required. Call .device() before .build()")

        if not self.__llm:
            raise ConfigurationError("LLM port is required. Call .llm() before .build()")

        # Apply defaults for optional ports
        if not self.__memory:
            self.__memory = SQLiteMemory(path_manager=self.__path_manager)

        if not self.__knowledge:
            self.__knowledge = SQLiteKnowledge(path_manager=self.__path_manager)

        if not self.__storage:
            self.__storage = LocalStorage(path_manager=self.__path_manager)

        if not self.__signal:
            self.__signal = NoopSignal()

        if not self.__telemetry:
            self.__telemetry = StructlogAdapter()

        return FathomRunner(
            llm=self.__llm,
            config=self.__config,
            memory=self.__memory,
            device=self.__device,
            signal=self.__signal,
            storage=self.__storage,
            knowledge=self.__knowledge,
            telemetry=self.__telemetry,
            path_manager=self.__path_manager,
        )


class Fathom:
    """
    Main entry point for Fathom library.
    """

    @staticmethod
    def builder(path_manager: Optional[SharedPathManager] = None) -> FathomBuilder:
        """
        Create a new builder instance.
        """

        return FathomBuilder(path_manager=path_manager)
