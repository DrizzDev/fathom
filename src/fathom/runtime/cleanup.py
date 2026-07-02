from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Iterable, Tuple

if TYPE_CHECKING:
    from logging import Logger


class ResourceCloser:
    """
    Best-effort closer for adapters that expose aclose/close/cleanup.
    """

    __METHODS: Tuple[str, ...] = ("aclose", "close", "cleanup")

    def __init__(self, *, logger: Logger, message: str) -> None:
        """
        Bind the logger and warning message used when a close call fails.
        """

        self.__logger = logger
        self.__message = message

    async def drain(self, *, resources: Iterable[Any]) -> None:
        """
        Close every resource in reverse construction order.
        """

        for resource in reversed(list(resources)):
            try:
                await self.__close_one(resource=resource)
            except Exception as exception:
                self.__logger.warning(f"{self.__message}: {exception}")

    async def __close_one(self, *, resource: Any) -> None:
        """
        Invoke the first available close method on one resource.
        """

        for method_name in self.__METHODS:
            close = getattr(resource, method_name, None)

            if close is None:
                continue

            result = close()
            if inspect.isawaitable(result):
                await result

            return
