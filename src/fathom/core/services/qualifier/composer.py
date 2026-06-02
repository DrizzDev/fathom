from __future__ import annotations

from typing import TYPE_CHECKING

from fathom.core.services.qualifier.factory import IntentQualifierFactory
from fathom.interfaces.factory import LLMFactoryPort
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.schemas.configuration import QualifierConfiguration

if TYPE_CHECKING:
    from fathom.runtime.assembly import RunAssemblyBuilder


class QualifierComposer:
    """
    Composes the qualifier port with a dedicated low-temperature LLM when enabled.
    """

    def __init__(
        self, *, assembly: "RunAssemblyBuilder", llm_factory: LLMFactoryPort
    ) -> None:
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
            qualifier_llm = self.__llm_factory.create(
                configuration=self.__assembly.build_qualifier_model_configuration(configuration=configuration)
            )

        return IntentQualifierFactory.create(llm=qualifier_llm, configuration=configuration)
