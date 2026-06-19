from __future__ import annotations

from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from fathom.constants import ActionType


class HITLCapability(BaseModel):
    """
    Capability flags governing human-in-the-loop interactions.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(description="Whether a human operator is available.")


class DeviceCapability(BaseModel):
    """
    Per-platform capability flags describing which system-level actions the underlying device adapter can dispatch.
    """

    model_config = ConfigDict(frozen=True)

    system_back_supported: bool = Field(
        default=True,
        description=(
            "Whether the device adapter supports a system back action. "
            "Android: True. iOS: False (the OS has no system back gesture)."
        ),
    )
    system_home_supported: bool = Field(
        default=True,
        description=(
            "Whether the device adapter supports a system home action. "
            "Defaults to True; reserved for future platforms that may differ."
        ),
    )
    system_scroll_supported: bool = Field(
        default=True,
        description=(
            "Whether the device adapter supports a generic system scroll "
            "gesture. Defaults to True on all current platforms."
        ),
    )

    def supports(self, *, action_type: ActionType) -> bool:
        """
        Return whether the device can dispatch the given system-level action.

        Action types not covered by an explicit flag are assumed supported
        (e.g. TAP, TYPE, SWIPE_* — these are app-level gestures every adapter
        implements). Only system-level chrome actions are gated here.
        """

        mapping: Mapping[ActionType, bool] = {
            ActionType.BACK: self.system_back_supported,
            ActionType.HOME: self.system_home_supported,
            ActionType.SCROLL: self.system_scroll_supported,
        }
        return mapping.get(action_type, True)


class RuntimeCapabilities(BaseModel):
    """
    Runtime capability flags injected at composition root.
    """

    model_config = ConfigDict(frozen=True)

    hitl: HITLCapability = Field(description="Human-in-the-loop capability flags.")

    device: DeviceCapability = Field(
        default_factory=DeviceCapability,
        description="Device-level capability flags consumed by recovery and loop logic.",
    )
