from __future__ import annotations

from fathom.core.services.qualifier.llm import LLMIntentQualifier
from fathom.core.services.qualifier.permissive import PermissiveIntentQualifier
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.qualifier import IntentQualifierPort
from fathom.schemas.configuration import QualifierConfiguration


class IntentQualifierFactory:
    """
    Selects the qualifier implementation that matches the supplied configuration.
    """

    @staticmethod
    def create(
        *, llm: LLMPort, configuration: QualifierConfiguration
    ) -> IntentQualifierPort:
        """
        Return the LLM-backed qualifier when enabled, otherwise the permissive one.
        """

        if not configuration.enabled:
            return PermissiveIntentQualifier()

        return LLMIntentQualifier(llm=llm, configuration=configuration)
