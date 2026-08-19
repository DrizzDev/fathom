from __future__ import annotations

from typing import Awaitable, Callable, Optional, Tuple

from fathom.constants.screen import HIERARCHY_BREAKER_COOLDOWN_CAPTURES, HierarchyProvenance
from fathom.schemas.screens import HierarchySnapshot


class HierarchyBreaker:
    """
    Bounds repeated hierarchy-dump failures for one perception adapter.
    """

    def __init__(self, *, cooldown: int = HIERARCHY_BREAKER_COOLDOWN_CAPTURES) -> None:
        """
        Bind the breaker to a bounded cooldown measured in captures.
        """

        self.__remaining = 0
        self.__cooldown = cooldown

    @property
    def open(self) -> bool:
        """
        Whether the breaker is currently skipping hierarchy dumps.
        """

        return self.__remaining > 0

    async def snapshot(
        self,
        *,
        screenshot: Callable[[], Awaitable[bytes]],
        dump: Callable[[], Awaitable[Tuple[bytes, Optional[str]]]],
    ) -> HierarchySnapshot:
        """
        Skip the dump while open, else attempt it and trip the breaker when it yields no hierarchy.
        """

        if self.open:
            self.__remaining -= 1
            return HierarchySnapshot(
                image=await screenshot(), provenance=HierarchyProvenance.CIRCUIT_OPEN
            )

        image, hierarchy = await dump()

        if hierarchy is None:
            self.__remaining = self.__cooldown
            return HierarchySnapshot(image=image, provenance=HierarchyProvenance.ATTEMPT_FAILED)

        return HierarchySnapshot(image=image, hierarchy=hierarchy)
