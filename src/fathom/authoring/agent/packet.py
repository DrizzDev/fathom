from __future__ import annotations

from fathom.schemas.authoring import AuthoringTask
from fathom.schemas.authoring.packet import AuthoringPacket
from fathom.schemas.authoring.reference import AuthoringDialectReference


class AuthoringPacketBuilder:
    """
    Builds typed prompt packets from authoring tasks and dialect references.
    """

    def build(self, *, task: AuthoringTask, dialect: AuthoringDialectReference) -> AuthoringPacket:
        """
        Compose the task and dialect reference into one packet.
        """

        return AuthoringPacket(task=task, dialect=dialect)
