from __future__ import annotations

from fathom.constants.authoring import AuthoringStatus
from fathom.interfaces.authoring import AuthoringPort
from fathom.schemas.authoring import AuthoringResponse, AuthoringTask


class AuthoringAgent:
    """
    Single script-authoring worker for run, step, and repair tasks.
    """

    async def author(
        self,
        *,
        task: AuthoringTask,
        authoring: AuthoringPort,
    ) -> AuthoringResponse:
        """
        Author one script task through the configured source.
        """

        response = await authoring.author(task=task)
        if response.status is AuthoringStatus.GENERATED and not response.has_script:
            return AuthoringResponse(
                status=AuthoringStatus.FAILED,
                reason="The run authoring source returned an empty script.",
            )

        return response
