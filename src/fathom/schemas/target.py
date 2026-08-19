from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TargetAuthority(BaseModel):
    """
    Authoritative target application for a run, or unbound when none was declared.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    package: Optional[str] = Field(
        default=None,
        description="Authoritative target package; None when no target was explicitly requested.",
    )

    @property
    def bound(self) -> bool:
        """
        Whether an authoritative target package is known.
        """

        return self.package is not None

    @classmethod
    def requested(cls, *, package: str) -> "TargetAuthority":
        """
        Bind authority to a package the caller explicitly requested.
        """

        return cls(package=package)

    @classmethod
    def unbound(cls) -> "TargetAuthority":
        """
        Return unbound authority, used when no target was requested.
        """

        return cls(package=None)
