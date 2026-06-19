from __future__ import annotations

from typing import Final, Tuple

from fathom.constants.tools import ToolName, TurnMode
from fathom.core.agent.tools.policies.hitl import HitlToolPolicy
from fathom.core.agent.tools.policies.mode import TurnModeToolPolicy
from fathom.core.agent.tools.policy import ToolPolicy

DEFAULT_TOOL_POLICIES: Final[Tuple[ToolPolicy, ...]] = (
    HitlToolPolicy(tool=ToolName.ASK_USER),
    TurnModeToolPolicy(tool=ToolName.VERIFY_GOAL, required_mode=TurnMode.VERIFY),
    TurnModeToolPolicy(tool=ToolName.VALIDATE_STATE, required_mode=TurnMode.VERIFY),
)
