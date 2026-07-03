from __future__ import annotations

from typing import Protocol

from fathom.schemas.authoring import AuthoringResponse, AuthoringTask


class AuthoringPort(Protocol):
    """
    Port for producing authored script output for a task.
    """

    async def author(self, *, task: AuthoringTask) -> AuthoringResponse:
        """
        Author script output for the supplied task.
        """

        ...
