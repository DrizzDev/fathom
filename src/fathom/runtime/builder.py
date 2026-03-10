from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fathom.adapters.knowledge.sqlite import SQLiteKnowledge
from fathom.adapters.memory.sqlite import SQLiteMemory
from fathom.adapters.signal.noop import NoopSignal
from fathom.adapters.storage.local import LocalStorage
from fathom.adapters.summarization.llm import LLMSummarizer
from fathom.adapters.telemetry.structlog import StructlogAdapter
from fathom.base.paths import SharedPathManager
from fathom.core.exceptions import ConfigurationError
from fathom.infrastructure.memory.ledger import Ledger
from fathom.infrastructure.memory.sqlite import SQLiteMemoryProvider
from fathom.schemas.configuration import (
    ExecutionConfiguration,
    ExplorationConfiguration,
    FathomConfiguration,
    IntentConfiguration,
)
from fathom.schemas.orchestration import RealignmentPolicy
from fathom.settings.env import FathomSettings

if TYPE_CHECKING:
    from fathom.interfaces.device import DevicePort
    from fathom.interfaces.knowledge import KnowledgePort
    from fathom.interfaces.llm import LLMPort
    from fathom.interfaces.memory import MemoryPort
    from fathom.interfaces.perception import PerceptionPort
    from fathom.interfaces.signal import SignalPort
    from fathom.interfaces.storage import StoragePort
    from fathom.interfaces.summarization import SummarizationPort
    from fathom.interfaces.telemetry import TelemetryPort
    from fathom.runtime.runner import FathomRunner


class FathomBuilder:
    """
    Fluent builder for Fathom configuration.

    Methods are order-independent. Validation happens at build().
    """

    def __init__(self, path_manager: Optional[SharedPathManager] = None) -> None:
        """
        Initialize builder with no ports configured.

        Args:
            path_manager: Optional shared path manager instance
        """

        self.__device: Optional[DevicePort] = None
        self.__perception: Optional[PerceptionPort] = None

        self.__llm: Optional[LLMPort] = None
        self.__memory: Optional[MemoryPort] = None
        self.__knowledge: Optional[KnowledgePort] = None

        self.__signal: Optional[SignalPort] = None
        self.__storage: Optional[StoragePort] = None
        self.__telemetry: Optional[TelemetryPort] = None
        self.__summarizer: Optional[SummarizationPort] = None
        self.__realignment: Optional[RealignmentPolicy] = None

        self.__path_manager = path_manager
        self.__config: FathomConfiguration = FathomConfiguration()

    def with_device(self, port: DevicePort) -> FathomBuilder:
        """
        Configure device port.

        Args:
            port: Device port implementation

        Returns:
            Builder instance for chaining
        """

        self.__device = port
        return self

    def with_llm(self, port: LLMPort) -> FathomBuilder:
        """
        Configure LLM port.

        Args:
            port: LLM port implementation

        Returns:
            Builder instance for chaining
        """

        self.__llm = port
        return self

    def with_perception(self, port: PerceptionPort) -> FathomBuilder:
        """
        Configure perception port.

        Args:
            port: Perception port implementation

        Returns:
            Builder instance for chaining
        """

        self.__perception = port
        return self

    def with_memory(self, port: MemoryPort) -> FathomBuilder:
        """
        Configure memory port.

        Args:
            port: Memory port implementation

        Returns:
            Builder instance for chaining
        """

        self.__memory = port
        return self

    def with_knowledge(self, port: KnowledgePort) -> FathomBuilder:
        """
        Configure knowledge port.

        Args:
            port: Knowledge port implementation

        Returns:
            Builder instance for chaining
        """

        self.__knowledge = port
        return self

    def with_signal(self, port: SignalPort) -> FathomBuilder:
        """
        Configure signal port.

        Args:
            port: Signal port implementation

        Returns:
            Builder instance for chaining
        """

        self.__signal = port
        return self

    def with_storage(self, port: StoragePort) -> FathomBuilder:
        """
        Configure storage port.

        Args:
            port: Storage port implementation

        Returns:
            Builder instance for chaining
        """

        self.__storage = port
        return self

    def with_telemetry(self, port: TelemetryPort) -> FathomBuilder:
        """
        Configure telemetry port.

        Args:
            port: Telemetry port implementation

        Returns:
            Builder instance for chaining
        """

        self.__telemetry = port
        return self

    def with_summarizer(self, port: SummarizationPort) -> FathomBuilder:
        """
        Configure summarization port.

        Args:
            port: Summarization port implementation

        Returns:
            Builder instance for chaining
        """

        self.__summarizer = port
        return self

    def with_config(self, configuration: FathomConfiguration) -> FathomBuilder:
        """
        Configure Fathom settings.

        Args:
            configuration: Complete Fathom configuration

        Returns:
            Builder instance for chaining
        """

        self.__config = configuration
        return self

    def with_execution_config(self, configuration: ExecutionConfiguration) -> FathomBuilder:
        """
        Configure execution engine settings.

        Args:
            configuration: Execution configuration

        Returns:
            Builder instance for chaining
        """

        self.__config.engine = configuration
        return self

    def with_intent_config(self, configuration: IntentConfiguration) -> FathomBuilder:
        """
        Configure intent strategy settings.

        Args:
            configuration: Intent configuration

        Returns:
            Builder instance for chaining
        """

        self.__config.intent = configuration
        return self

    def with_exploration_config(self, configuration: ExplorationConfiguration) -> FathomBuilder:
        """
        Configure exploration strategy settings.

        Args:
            configuration: Exploration configuration

        Returns:
            Builder instance for chaining
        """

        self.__config.exploration = configuration
        return self

    def with_realignment(self, policy: RealignmentPolicy) -> FathomBuilder:
        """
        Configure realignment policy.

        Args:
            policy: Realignment policy instance

        Returns:
            Builder instance for chaining
        """

        self.__realignment = policy
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

        from fathom.runtime.runner import FathomRunner

        if not self.__path_manager:
            self.__path_manager = SharedPathManager(settings=FathomSettings())

        if not self.__device:
            raise ConfigurationError("Device port is required. Call .with_device() before .build()")

        if not self.__llm:
            raise ConfigurationError("LLM port is required. Call .with_llm() before .build()")

        if not self.__perception:
            raise ConfigurationError(
                "Perception port is required. Call .with_perception() before .build()"
            )

        if not self.__memory:
            ledger = Ledger(database_path=self.__path_manager.get_ledger_db_path())
            provider = SQLiteMemoryProvider(
                database_path=self.__path_manager.get_knowledge_db_path()
            )
            self.__memory = SQLiteMemory(ledger=ledger, provider=provider)

        if not self.__knowledge:
            self.__knowledge = SQLiteKnowledge(path_manager=self.__path_manager)

        if not self.__signal:
            self.__signal = NoopSignal()

        if not self.__storage:
            self.__storage = LocalStorage(path_manager=self.__path_manager)

        if not self.__telemetry:
            self.__telemetry = StructlogAdapter()

        if not self.__summarizer:
            self.__summarizer = LLMSummarizer(llm=self.__llm)

        if not self.__realignment:
            self.__realignment = RealignmentPolicy()

        return FathomRunner(
            llm=self.__llm,
            device=self.__device,
            perception=self.__perception,
            config=self.__config,
            memory=self.__memory,
            signal=self.__signal,
            storage=self.__storage,
            knowledge=self.__knowledge,
            telemetry=self.__telemetry,
            summarizer=self.__summarizer,
            realignment=self.__realignment,
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

        Args:
            path_manager: Optional shared path manager

        Returns:
            New FathomBuilder instance
        """

        return FathomBuilder(path_manager=path_manager)
