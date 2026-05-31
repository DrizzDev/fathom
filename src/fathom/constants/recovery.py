from typing import Final, FrozenSet

from fathom.schemas.vision import ActionKind

AUTONOMOUS_RECOVERY_ACTIVE_KINDS: Final[FrozenSet[ActionKind]] = frozenset(
    {
        ActionKind.INPUT,
        ActionKind.NAVIGATION,
    }
)
