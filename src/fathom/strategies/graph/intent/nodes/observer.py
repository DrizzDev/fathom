from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, List, Optional

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.runtime import (
    DEFAULT_LOCAL_PERCEPTION_BUDGET,
    DEFAULT_LOCALIZATION_BUDGET,
    DEFAULT_OCR_PERCEPTION_BUDGET,
)
from fathom.constants.state import CommonStateKey
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.observation import ScreenObservation
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle, ScreenState
from fathom.schemas.ui import LabeledElement
from fathom.strategies.graph.state import IntentGraphState

if TYPE_CHECKING:
    from fathom.strategies.graph.context import GraphContext


class ScreenObserver:
    """
    Builds runtime screen state and observation evidence for graph nodes.
    """

    def __init__(self, *, context: GraphContext) -> None:
        """
        Initialize the observer with the shared graph context.
        """

        self.__context = context

    def build_screen_state(
        self,
        *,
        visual_hash: str,
        capture: ScreenCapture,
        xml_hash: Optional[str] = None,
        interaction_hash: Optional[str] = None,
    ) -> ScreenState:
        """
        Build a normalized :class:`ScreenState` from the available capture signals.
        """

        return ScreenState(
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash=hashlib.md5(
                capture.activity.encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:VISUAL_HASH_LENGTH],
            visual_hash=visual_hash,
            xml_hash=xml_hash,
            interaction_hash=interaction_hash,
        )

    def resolve_capture_hashes(
        self,
        *,
        capture: ScreenCapture,
        elements: Optional[List[LabeledElement]] = None,
    ) -> ScreenHashBundle:
        """
        Compute the visual, XML, and interaction hashes for one capture.
        """

        return ScreenHashBundle(
            xml_hash=self.__context.perception.compute_xml_hash(capture=capture),
            visual_hash=self.__context.perception.compute_visual_hash(capture=capture),
            interaction_hash=self.__context.perception.compute_interaction_hash(elements=elements),
        )

    async def observe(
        self,
        *,
        capture: ScreenCapture,
        hashes: ScreenHashBundle,
        elements: List[LabeledElement],
    ) -> ScreenObservation:
        """
        Build the runtime screen observation for localization and verification.
        """

        return await self.__context.screen_observer.observe(
            hashes=hashes,
            capture=capture,
            manifest=tuple(elements),
            budget=self.__budget(),
            session_id=self.__context.workflow_id,
            step_number=self.__context.agent_state.step_count,
        )

    async def fallback_observation(
        self,
        *,
        state: IntentGraphState,
        capture: ScreenCapture,
    ) -> ScreenObservation:
        """
        Return a valid observation when the graph state lacks one.
        """

        observation = state.get(CommonStateKey.SCREEN_OBSERVATION)
        if isinstance(observation, ScreenObservation):
            return observation

        hashes = ScreenHashBundle(
            visual_hash=(
                capture.state.visual_hash
                if capture.state is not None
                else self.__context.perception.compute_visual_hash(capture=capture)
            ),
            xml_hash=self.__context.perception.compute_xml_hash(capture=capture),
            interaction_hash=(
                capture.state.interaction_hash
                if capture.state is not None and capture.state.interaction_hash is not None
                else ""
            ),
        )
        return await self.__context.screen_observer.observe(
            manifest=(),
            hashes=hashes,
            capture=capture,
            budget=self.__budget(),
            session_id=self.__context.workflow_id,
            step_number=self.__context.agent_state.step_count,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        """
        Return the default perception budget used by the observer paths.
        """

        return PerceptionBudget(
            ocr=DEFAULT_OCR_PERCEPTION_BUDGET,
            local=DEFAULT_LOCAL_PERCEPTION_BUDGET,
            localization=DEFAULT_LOCALIZATION_BUDGET,
        )
