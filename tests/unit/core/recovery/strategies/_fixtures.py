from __future__ import annotations

from typing import List, Optional, Tuple

from fathom.core.recovery.types import RecoveryRequest, RecoveryTrigger
from fathom.schemas.actions import Bounds
from fathom.schemas.escape import EscapeCategory, EscapeReport
from fathom.schemas.localization import LocalizationCandidate
from fathom.schemas.observation import (
    ElementRole,
    ElementSource,
    KeyboardObservation,
    OverlayObservation,
    PerceivedElement,
    ScreenObservation,
    ScrollRegion,
)
from fathom.schemas.screens import ScreenCapture, ScreenHashBundle
from fathom.schemas.supervision import BlockReason


def capture() -> ScreenCapture:
    """
    Return a minimal screen capture for recovery request fixtures.
    """

    return ScreenCapture(
        width=1206,
        height=2622,
        activity="bundl.swiggy.production",
        image=b"\x89PNG\r\n\x1a\n",
        timestamp=0,
    )


def hashes() -> ScreenHashBundle:
    """
    Return a minimal screen hash bundle for observation fixtures.
    """

    return ScreenHashBundle(
        visual_hash="0" * 16,
        xml_hash="a" * 16,
        interaction_hash="b" * 16,
    )


def bounds(x: int = 0, y: int = 0, width: int = 60, height: int = 30) -> Bounds:
    """
    Return a Bounds value with sensible defaults for fixture elements.
    """

    return Bounds(x=x, y=y, width=width, height=height)


def element(
    *,
    identifier: str,
    text: Optional[str] = None,
    role: ElementRole = ElementRole.BUTTON,
    source: ElementSource = ElementSource.XML,
    tappable: bool = True,
    rect: Optional[Bounds] = None,
) -> PerceivedElement:
    """
    Build a perceived element with optional bounds override.
    """

    return PerceivedElement(
        identifier=identifier,
        text=text,
        role=role,
        source=source,
        tappable=tappable,
        confidence=0.9,
        bounds=rect or bounds(),
    )


def observation(
    *,
    keyboard: Optional[KeyboardObservation] = None,
    overlays: Tuple[OverlayObservation, ...] = (),
    elements: Tuple[PerceivedElement, ...] = (),
    calls_to_action: Tuple[PerceivedElement, ...] = (),
    scroll: Tuple[ScrollRegion, ...] = (),
) -> ScreenObservation:
    """
    Build a screen observation with the supplied subset of fields.
    """

    return ScreenObservation(
        activity="bundl.swiggy.production",
        hashes=hashes(),
        elements=elements,
        overlays=overlays,
        keyboard=keyboard or KeyboardObservation(visible=False),
        scroll=scroll,
        calls_to_action=calls_to_action,
    )


def candidate(
    *,
    reason: str,
    score: float = 0.7,
    matched: Optional[PerceivedElement] = None,
) -> LocalizationCandidate:
    """
    Build a localization candidate referencing the supplied element.
    """

    return LocalizationCandidate(
        element=matched,
        point=None,
        score=score,
        reason=reason,
    )


def request(
    *,
    trigger: RecoveryTrigger = RecoveryTrigger.NO_PROGRESS,
    block_reason: Optional[BlockReason] = None,
    stuck_sub_goal: str = "Tap on Continue",
    reason: str = "fixture reason",
    pending: Optional[List[str]] = None,
    recent: Optional[List[str]] = None,
    candidates: Optional[List[LocalizationCandidate]] = None,
    screen: Optional[ScreenObservation] = None,
    escape_report: Optional[EscapeReport] = None,
) -> RecoveryRequest:
    """
    Build a RecoveryRequest with the supplied context overrides.
    """

    return RecoveryRequest(
        trigger=trigger,
        capture=capture(),
        reason=reason,
        hint=None,
        stuck_sub_goal=stuck_sub_goal,
        pending_sub_goals=pending or [],
        recent_actions=recent or [],
        escape_report=escape_report,
        block_reason=block_reason,
        observation=screen,
        candidates=candidates or [],
    )


def escape(category: EscapeCategory, detail: str = "fixture detail") -> EscapeReport:
    """
    Build an EscapeReport with the supplied category.
    """

    return EscapeReport(category=category, detail=detail)
