from __future__ import annotations

import io
import math
from logging import getLogger
from typing import Optional, Tuple, cast

from PIL import Image, ImageDraw, ImageFont

from fathom.constants.drawing import SourceColor, TraceDrawing
from fathom.core.artifact.drawing import BoxDrawer
from fathom.interfaces.artifact import ArtifactRendererPort
from fathom.schemas.artifact import (
    ArtifactKind,
    ArtifactRecord,
    CvPerceptionPayload,
    HierarchyXmlPayload,
    IconPerceptionPayload,
    OcrPerceptionPayload,
    OcrRawPayload,
    OverlayPerceptionPayload,
    PerceptionPayload,
    ScreenshotPayload,
    ScriptPayload,
    TracePayload,
    VerificationPayload,
    VisionPerceptionPayload,
)
from fathom.schemas.observation import ElementSource, PerceivedElement, ScreenObservation
from fathom.schemas.screens import ScreenCapture
from fathom.utils.coordinates import CoordinateConverter

logger = getLogger(name=__name__)


class PassthroughRenderer(ArtifactRendererPort):
    """
    Renderer for kinds whose payloads are already in final byte form.

    Handles :class:`ScreenshotPayload`, :class:`HierarchyXmlPayload`, :class:`ScriptPayload`, and
    :class:`AnnotatedPayload` (the capture's ``annotated_image`` is already PNG bytes upstream).
    """

    def __init__(self, *, kind: ArtifactKind) -> None:
        """
        Bind this renderer to one specific :class:`ArtifactKind`.
        """

        self.__kind = kind

    @property
    def kind(self) -> ArtifactKind:
        """
        Stable identity of the kind this renderer handles.
        """

        return self.__kind

    def render(self, *, record: ArtifactRecord) -> bytes:
        """
        Extract the already-rendered bytes from the typed payload.
        """

        payload = record.payload
        if isinstance(payload, ScreenshotPayload):
            return payload.capture.image
        if isinstance(payload, HierarchyXmlPayload):
            return payload.content.encode("utf-8")
        if isinstance(payload, (ScriptPayload, OcrRawPayload)):
            return payload.content.encode("utf-8")
        return self.__resolve_annotated_image(record=record)

    def __resolve_annotated_image(self, *, record: ArtifactRecord) -> bytes:
        """
        Return the annotated bytes for :class:`AnnotatedPayload` records,
        falling back to the raw capture when no annotation was attached.
        """

        capture = record.payload.capture  # type: ignore[union-attr]
        if capture.annotated_image is not None:
            return capture.annotated_image
        return capture.image


class PerceptionRenderer(ArtifactRendererPort):
    """
    Renders the merged-perception debug image.

    Draws every :class:`PerceivedElement`, plus overlay rectangles and call-to-action boxes,
    through the shared :class:`BoxDrawer`, which owns colour, font, outline, and label formatting.
    """

    def __init__(self, *, drawer: Optional[BoxDrawer] = None) -> None:
        """
        Bind this renderer to a shared :class:`BoxDrawer`.
        """

        self.__drawer = drawer or BoxDrawer()

    @property
    def kind(self) -> ArtifactKind:
        """
        Stable identity of the kind this renderer handles.
        """

        return ArtifactKind.PERCEPTION

    def render(self, *, record: ArtifactRecord) -> bytes:
        """
        Compose every perception layer onto a copy of the source capture.
        """

        payload = cast("PerceptionPayload", record.payload)
        canvas = Image.open(io.BytesIO(payload.capture.image)).convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")

        for element in payload.observation.elements:
            self.__draw_element(draw=draw, element=element)

        for overlay in payload.observation.overlays:
            self.__draw_rect(
                draw=draw,
                bounds=(
                    overlay.bounds.x,
                    overlay.bounds.y,
                    overlay.bounds.x + overlay.bounds.width,
                    overlay.bounds.y + overlay.bounds.height,
                ),
                color=SourceColor.OVERLAY,
                label="overlay",
            )

        for call in payload.observation.calls_to_action:
            self.__draw_rect(
                draw=draw,
                bounds=(
                    call.bounds.x,
                    call.bounds.y,
                    call.bounds.x + call.bounds.width,
                    call.bounds.y + call.bounds.height,
                ),
                color=SourceColor.CALL_TO_ACTION,
                label=call.text or "cta",
            )

        return self.__export(canvas=canvas)

    def __draw_element(self, *, draw: ImageDraw.ImageDraw, element: PerceivedElement) -> None:
        """
        Hand one :class:`PerceivedElement` to the shared box drawer.
        """

        self.__drawer.draw(
            canvas=draw,
            bounds=(
                element.bounds.x,
                element.bounds.y,
                element.bounds.x + element.bounds.width,
                element.bounds.y + element.bounds.height,
            ),
            source=element.source,
            label_id=element.label_id,
            text=element.text,
            role=element.role.value,
        )

    def __draw_rect(
        self,
        *,
        draw: ImageDraw.ImageDraw,
        bounds: Tuple[int, int, int, int],
        color: str,
        label: str,
    ) -> None:
        """
        Draw a colour-overridden rectangle (overlays / call-to-action).
        """

        self.__drawer.draw(
            canvas=draw,
            bounds=bounds,
            source=ElementSource.XML,
            text=label,
            color=color,
        )

    @staticmethod
    def __export(*, canvas: Image.Image) -> bytes:
        """
        Encode the rendered canvas as PNG bytes.
        """

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()


class SourceFilteredPerceptionRenderer(ArtifactRendererPort):
    """
    Renders a single-source slice of the observation.

    Parameterised by :class:`ArtifactKind` (routing discriminator) and :class:`ElementSource`
    (projection filter) so OCR / CV / icon / vision images share one primitive. Overlay-only
    artifacts go through :class:`OverlayPerceptionRenderer` since overlays are not perceived elements.
    """

    def __init__(
        self,
        *,
        kind: ArtifactKind,
        source: ElementSource,
        drawer: Optional[BoxDrawer] = None,
    ) -> None:
        """
        Bind the renderer to its kind / source / shared drawer.
        """

        self.__kind = kind
        self.__source = source
        self.__drawer = drawer or BoxDrawer()

    @property
    def kind(self) -> ArtifactKind:
        """
        Stable identity of the kind this renderer handles.
        """

        return self.__kind

    def render(self, *, record: ArtifactRecord) -> bytes:
        """
        Compose source-filtered boxes onto a copy of the source capture.
        """

        payload = cast(
            "OcrPerceptionPayload | CvPerceptionPayload | IconPerceptionPayload | VisionPerceptionPayload",
            record.payload,
        )
        canvas = Image.open(io.BytesIO(payload.capture.image)).convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")

        for element in self.__filtered(observation=payload.observation):
            self.__drawer.draw(
                canvas=draw,
                bounds=(
                    element.bounds.x,
                    element.bounds.y,
                    element.bounds.x + element.bounds.width,
                    element.bounds.y + element.bounds.height,
                ),
                source=element.source,
                label_id=element.label_id,
                text=element.text,
                role=element.role.value,
            )

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()

    def __filtered(self, *, observation: ScreenObservation) -> Tuple[PerceivedElement, ...]:
        """
        Project to this source; drop tokens covered by a line/paragraph of the same source.
        """

        same_source = tuple(
            element for element in observation.elements if element.source is self.__source
        )

        phrases = tuple(
            element for element in same_source if self.__is_phrase_identifier(element=element)
        )
        if not phrases:
            return same_source

        return tuple(
            element
            for element in same_source
            if element in phrases or not self.__contained_in_any(element=element, parents=phrases)
        )

    @staticmethod
    def __is_phrase_identifier(*, element: PerceivedElement) -> bool:
        """
        Return whether ``element`` is a row- or paragraph-level OCR phrase.
        """

        identifier = element.identifier or ""
        return "_line_" in identifier or "_paragraph_" in identifier

    @classmethod
    def __contained_in_any(
        cls, *, element: PerceivedElement, parents: Tuple[PerceivedElement, ...]
    ) -> bool:
        """
        Return whether the element's bounds are fully inside any parent's bounds.
        """

        for parent in parents:
            if parent is element:
                continue

            if cls.__contains(outer=parent, inner=element):
                return True

        return False

    @staticmethod
    def __contains(*, outer: PerceivedElement, inner: PerceivedElement) -> bool:
        """
        Return whether ``inner``'s bounds lie inside ``outer``'s bounds.
        """

        return (
            outer.bounds.x <= inner.bounds.x
            and outer.bounds.y <= inner.bounds.y
            and outer.bounds.x + outer.bounds.width >= inner.bounds.x + inner.bounds.width
            and outer.bounds.y + outer.bounds.height >= inner.bounds.y + inner.bounds.height
        )


class OverlayPerceptionRenderer(ArtifactRendererPort):
    """
    Renders the overlay-only debug image.

    Projects :class:`OverlayObservation` rectangles from the screen observation; perceived elements are not drawn.
    Overlay shape is distinct from per-element rendering so this renderer cannot share :class:`SourceFilteredPerceptionRenderer`.
    """

    def __init__(self, *, drawer: Optional[BoxDrawer] = None) -> None:
        """
        Bind the renderer to a shared :class:`BoxDrawer`.
        """

        self.__drawer = drawer or BoxDrawer()

    @property
    def kind(self) -> ArtifactKind:
        """
        Stable identity of the kind this renderer handles.
        """

        return ArtifactKind.OVERLAY_PERCEPTION

    def render(self, *, record: ArtifactRecord) -> bytes:
        """
        Draw every detected overlay rectangle on the source capture.
        """

        payload = cast("OverlayPerceptionPayload", record.payload)
        canvas = Image.open(io.BytesIO(payload.capture.image)).convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")

        for overlay in payload.observation.overlays:
            self.__drawer.draw(
                canvas=draw,
                bounds=(
                    overlay.bounds.x,
                    overlay.bounds.y,
                    overlay.bounds.x + overlay.bounds.width,
                    overlay.bounds.y + overlay.bounds.height,
                ),
                source=ElementSource.XML,
                text="overlay",
                color=SourceColor.OVERLAY,
            )

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()


class TraceRenderer(ArtifactRendererPort):
    """
    Renders the post-action trace image.

    Draws the executed action's coordinates on the pre-action capture
    (tap target circle for point actions; arrow for swipe / scroll).
    """

    __POINT_ACTIONS: Tuple[str, ...] = ("tap", "type", "long_press")
    __SWIPE_ACTIONS: Tuple[str, ...] = (
        "swipe",
        "scroll",
        "swipe_up",
        "swipe_down",
        "swipe_left",
        "swipe_right",
    )

    @property
    def kind(self) -> ArtifactKind:
        """
        Stable identity of the kind this renderer handles.
        """

        return ArtifactKind.TRACE

    def render(self, *, record: ArtifactRecord) -> bytes:
        """
        Draw the action overlay on a copy of the source capture.
        """

        payload = cast("TracePayload", record.payload)
        canvas = Image.open(io.BytesIO(payload.capture.image)).convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")

        projected = self.__project_coords(
            coords=payload.coords,
            capture=payload.capture,
            canvas_size=canvas.size,
            session_id=record.session_id,
            action_type=payload.action.action_type.value,
        )

        action_type = payload.action.action_type.value

        if action_type in self.__POINT_ACTIONS and len(projected) >= 2:
            self.__draw_point(draw=draw, coords=projected)

        elif action_type in self.__SWIPE_ACTIONS and len(projected) >= 4:
            self.__draw_swipe(draw=draw, coords=projected)

        self.__draw_label(draw=draw, label=payload.action.to_description())

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def __project_coords(
        *,
        session_id: str,
        action_type: str,
        capture: ScreenCapture,
        coords: Tuple[int, ...],
        canvas_size: Tuple[int, int],
    ) -> Tuple[int, ...]:
        """
        Project logical dispatch coords through the capture's pixel canvas; raw coords are returned on invalid dims.
        Fail-clean emits ``renderer.trace.unscaled`` so a missing/invalid scale never silently invents a factor.
        """

        pixel_width, pixel_height = canvas_size

        logical_width = capture.width
        logical_height = capture.height

        if (
            logical_width <= 0
            or logical_height <= 0
            or pixel_width <= 0
            or pixel_height <= 0
            or pixel_width < logical_width
            or pixel_height < logical_height
        ):
            logger.warning(
                "Trace renderer drawing coords without projection",
                extra={
                    "workflow.id": session_id,
                    "action.type": action_type,
                    "coords.raw": list(coords),
                    "event": "renderer.trace.unscaled",
                    "component": "core.artifact.renderer",
                    "capture.pixel": {"width": pixel_width, "height": pixel_height},
                    "capture.logical": {"width": logical_width, "height": logical_height},
                },
            )
            return coords

        converter = CoordinateConverter(
            workflow_id=session_id,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            logical_width=logical_width,
            logical_height=logical_height,
        )
        projected: Tuple[int, ...] = tuple(
            value
            for index in range(0, len(coords) - 1, 2)
            for value in converter.capture_point(x=coords[index], y=coords[index + 1])
        )
        logger.info(
            "Trace renderer projected coords",
            extra={
                "workflow.id": session_id,
                "action.type": action_type,
                "coords.logical": list(coords),
                "coords.pixel": list(projected),
                "event": "renderer.trace.projected",
                "component": "core.artifact.renderer",
                "capture.pixel": {"width": pixel_width, "height": pixel_height},
                "capture.logical": {"width": logical_width, "height": logical_height},
            },
        )
        return projected

    @staticmethod
    def __draw_point(*, draw: ImageDraw.ImageDraw, coords: Tuple[int, ...]) -> None:
        """
        Render the tap-target circle and centre dot.
        """

        x, y = coords[0], coords[1]
        radius = TraceDrawing.TAP_RADIUS

        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=SourceColor.ACTION,
            width=TraceDrawing.LINE_WIDTH // 2,
            fill=(255, 59, 48, 100),
        )

        centre = TraceDrawing.CENTER_DOT_RADIUS
        draw.ellipse(
            [x - centre, y - centre, x + centre, y + centre],
            fill=SourceColor.ACTION,
        )

    @staticmethod
    def __draw_swipe(*, draw: ImageDraw.ImageDraw, coords: Tuple[int, ...]) -> None:
        """
        Render the swipe / scroll arrow.
        """

        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

        draw.line([x1, y1, x2, y2], fill=SourceColor.ACTION, width=TraceDrawing.LINE_WIDTH)
        start_radius = TraceDrawing.SWIPE_START_RADIUS

        draw.ellipse(
            [x1 - start_radius, y1 - start_radius, x1 + start_radius, y1 + start_radius],
            fill=SourceColor.ACTION,
        )

        arrow_head = TraceRenderer.__arrow_head(start=(x1, y1), end=(x2, y2))
        if arrow_head is None:
            return

        draw.polygon(
            [(x2, y2), arrow_head[0], arrow_head[1]],
            fill=SourceColor.ACTION,
        )

    @staticmethod
    def __arrow_head(
        *, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Resolve the two base points for a directional arrowhead at the swipe endpoint.
        """

        end_x, end_y = end
        start_x, start_y = start

        delta_x = end_x - start_x
        delta_y = end_y - start_y

        length = math.hypot(delta_x, delta_y)
        if length == 0:
            return None

        unit_x = delta_x / length
        unit_y = delta_y / length

        perpendicular_x = -unit_y
        perpendicular_y = unit_x

        angle = math.radians(TraceDrawing.ARROW_HEAD_ANGLE_DEGREES)
        head_length = TraceDrawing.ARROW_HEAD_LENGTH

        back_x = end_x - unit_x * head_length * math.cos(angle)
        back_y = end_y - unit_y * head_length * math.cos(angle)
        offset_x = perpendicular_x * head_length * math.sin(angle)
        offset_y = perpendicular_y * head_length * math.sin(angle)

        return (
            (round(back_x + offset_x), round(back_y + offset_y)),
            (round(back_x - offset_x), round(back_y - offset_y)),
        )

    @staticmethod
    def __draw_label(*, draw: ImageDraw.ImageDraw, label: str) -> None:
        """
        Render the action description in the top-left corner.
        """

        font = ImageFont.load_default()

        draw.text(
            (10, 10),
            f"Action: {label}",
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
        )


class VerificationRenderer(ArtifactRendererPort):
    """
    Renders the verifier-stage artifact.

    Draws the verifier's verdict (complete / reason) as a labelled
    banner across the bottom of the inspected capture.
    """

    __VERDICT_HEIGHT: int = 80
    __PASS_COLOR: str = "#10B981"
    __FAIL_COLOR: str = "#EF4444"

    @property
    def kind(self) -> ArtifactKind:
        """
        Stable identity of the kind this renderer handles.
        """

        return ArtifactKind.VERIFICATION

    def render(self, *, record: ArtifactRecord) -> bytes:
        """
        Compose the verdict banner over the inspected capture.
        """

        payload = cast("VerificationPayload", record.payload)
        canvas = Image.open(io.BytesIO(payload.capture.image)).convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")

        width, height = canvas.size
        color = self.__PASS_COLOR if payload.verdict.complete else self.__FAIL_COLOR

        draw.rectangle(
            [0, height - self.__VERDICT_HEIGHT, width, height],
            fill=(255, 255, 255, 200),
            outline=color,
            width=4,
        )
        font = ImageFont.load_default()
        verdict_text = "VERIFIED" if payload.verdict.complete else "REJECTED"
        draw.text(
            (16, height - self.__VERDICT_HEIGHT + 12),
            f"{verdict_text}: {payload.verdict.reason[:80]}",
            font=font,
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")

        return buffer.getvalue()
