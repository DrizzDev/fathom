from __future__ import annotations

from typing import Optional

from fathom.schemas.screens import ScreenCapture, ScreenHashBundle, ScreenState


class ScreenFixtures:
    """
    Factory for :class:`ScreenState`, :class:`ScreenCapture`, and
    :class:`ScreenHashBundle` instances used across unit tests.
    """

    DEFAULT_ACTIVITY = "com.example.app"
    DEFAULT_VISUAL_HASH = "b" * 16
    DEFAULT_ACTIVITY_HASH = "a" * 16
    DEFAULT_XML_HASH = "c" * 16
    DEFAULT_INTERACTION_HASH = "d" * 16

    @classmethod
    def state(
        cls,
        *,
        activity: str = DEFAULT_ACTIVITY,
        timestamp: int = 0,
        visual_hash: str = DEFAULT_VISUAL_HASH,
        activity_hash: str = DEFAULT_ACTIVITY_HASH,
        xml_hash: Optional[str] = None,
        interaction_hash: Optional[str] = None,
    ) -> ScreenState:
        """
        Build a minimal :class:`ScreenState` for tests needing a stable identity.
        """

        return ScreenState(
            activity=activity,
            timestamp=timestamp,
            visual_hash=visual_hash,
            activity_hash=activity_hash,
            xml_hash=xml_hash,
            interaction_hash=interaction_hash,
        )

    @classmethod
    def capture(
        cls,
        *,
        activity: str = DEFAULT_ACTIVITY,
        width: int = 100,
        height: int = 200,
        timestamp: int = 1,
        image: bytes = b"png",
        xml_content: Optional[str] = None,
    ) -> ScreenCapture:
        """
        Build a minimal :class:`ScreenCapture` for planner / perception tests.
        """

        return ScreenCapture(
            activity=activity,
            width=width,
            height=height,
            image=image,
            timestamp=timestamp,
            xml_content=xml_content,
        )

    @classmethod
    def hash_bundle(
        cls,
        *,
        visual_hash: str = "0" * 16,
        xml_hash: str = DEFAULT_XML_HASH,
        interaction_hash: str = DEFAULT_INTERACTION_HASH,
    ) -> ScreenHashBundle:
        """
        Build a deterministic :class:`ScreenHashBundle` for observation-only tests.
        """

        return ScreenHashBundle(
            visual_hash=visual_hash,
            xml_hash=xml_hash,
            interaction_hash=interaction_hash,
        )
