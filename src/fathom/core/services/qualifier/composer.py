from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

from fathom.core.services.qualifier.factory import IntentQualifierFactory
from fathom.interfaces.factory import LLMFactoryPort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.schemas.configuration import LLMConfiguration, QualifierConfiguration

if TYPE_CHECKING:
    from fathom.runtime.assembly import RunAssemblyBuilder

logger = getLogger(__name__)


class QualifierComposer:
    """
    Composes the qualifier port with a dedicated low-temperature LLM when enabled.
    """

    def __init__(self, *, assembly: "RunAssemblyBuilder", llm_factory: LLMFactoryPort) -> None:
        """
        Wire the composer with the assembly that owns settings and the LLM factory.
        """

        self.__assembly = assembly
        self.__llm_factory = llm_factory

    def compose(
        self, *, planner_llm: LLMPort, configuration: QualifierConfiguration
    ) -> IntentQualifierPort:
        """
        Return the qualifier port for the given configuration, reusing planner_llm when disabled.
        """

        qualifier_llm = planner_llm

        if configuration.enabled:
            qualifier_llm = self.__construct_dedicated_llm(configuration=configuration)

        qualifier = IntentQualifierFactory.create(llm=qualifier_llm, configuration=configuration)
        logger.info(
            "qualifier.composed",
            extra={
                "impl": type(qualifier).__name__,
                "enabled": configuration.enabled,
                "model": qualifier_llm.model_name,
                "temperature": configuration.temperature,
                "thinking_level": configuration.thinking_level,
            },
        )
        return qualifier

    def __construct_dedicated_llm(self, *, configuration: QualifierConfiguration) -> LLMPort:
        """
        Build the dedicated low-temperature LLM, logging construction failures before re-raising.
        """

        llm_configuration: LLMConfiguration = self.__assembly.build_qualifier_model_configuration(
            configuration=configuration
        )
        try:
            return self.__llm_factory.create(configuration=llm_configuration)
        except Exception as exception:
            logger.warning(
                "qualifier.dedicated_llm_construction_failed",
                extra={
                    "reason": str(exception),
                    "model": llm_configuration.model,
                    "location": llm_configuration.location,
                    "project_id": llm_configuration.project_id,
                    "has_api_key": llm_configuration.api_key is not None,
                    "has_credentials": llm_configuration.credentials is not None,
                },
            )
            raise
