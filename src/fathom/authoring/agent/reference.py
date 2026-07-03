from __future__ import annotations

from fathom.constants.dialect import DialectName
from fathom.core.exceptions import ConfigurationError
from fathom.schemas.authoring.reference import (
    DRIZZ_COMMANDS,
    DRIZZ_GUIDE,
    AuthoringDialectReference,
)


class AuthoringReferenceProvider:
    """
    Supplies dialect references for authoring prompts.
    """

    def reference(self, *, dialect: DialectName) -> AuthoringDialectReference:
        """
        Return the prompt reference for a supported dialect.
        """

        if dialect is DialectName.DRIZZ:
            return AuthoringDialectReference(
                name=dialect,
                guide=DRIZZ_GUIDE,
                commands=DRIZZ_COMMANDS,
            )

        raise ConfigurationError(f"Unsupported authoring dialect '{dialect.value}'.")
