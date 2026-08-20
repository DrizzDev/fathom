from __future__ import annotations

from abc import ABC, abstractmethod

from fathom.schemas.artifact import (
    ArtifactKind,
    ArtifactMetadata,
    ArtifactReceipt,
    ArtifactRecord,
)


class ArtifactRendererPort(ABC):
    """
    Strategy that turns one typed :class:`ArtifactRecord` into byte form.

    One concrete renderer per :class:`ArtifactKind`. Adding a new kind means adding one renderer
    and registering it at the composition root; no existing renderer needs to change.
    """

    @property
    @abstractmethod
    def kind(self) -> ArtifactKind:
        """
        Stable identity of the kind this renderer handles.
        """

        raise NotImplementedError

    @abstractmethod
    def render(self, *, record: ArtifactRecord) -> bytes:
        """
        Return the artifact's final byte form ready for persistence.
        """

        raise NotImplementedError


class ArtifactSinkPort(ABC):
    """
    Persistence boundary for artifact bytes already staged onto local disk by the pipeline.

    Sinks receive the metadata slice, never the bytes-heavy typed payload, so the boundary stays
    cheap to serialize and one contract covers image, XML, and text artifacts alike.
    """

    @abstractmethod
    async def persist(
        self,
        *,
        content: bytes,
        metadata: ArtifactMetadata,
    ) -> ArtifactReceipt:
        """
        Persist the rendered bytes and report whether local cleanup is safe.
        """

        raise NotImplementedError
