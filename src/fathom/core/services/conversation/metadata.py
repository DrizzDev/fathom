from __future__ import annotations

from typing import Optional

from fathom.core.exceptions import InteractionError
from fathom.schemas.conversation import ThreadMetadataView, ThreadTitleMetadataView
from fathom.schemas.interaction import Thread


class ThreadMetadataProjector:
    """
    Projects stored thread metadata into the public conversation schema.
    """

    def view(self, *, thread: Thread) -> ThreadMetadataView:
        """
        Return typed public metadata for a stored thread.
        """

        return ThreadMetadataView(title=self.__title(thread=thread))

    def __title(self, *, thread: Thread) -> Optional[ThreadTitleMetadataView]:
        """
        Return typed title metadata when the stored thread includes it.
        """

        value = thread.metadata.entries.get("title")

        if value is None:
            return None

        if not isinstance(value, dict):
            raise InteractionError("Thread title metadata is invalid.")

        return ThreadTitleMetadataView.model_validate(value)
