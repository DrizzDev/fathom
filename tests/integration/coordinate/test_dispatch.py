"""
End-to-end integration tests for coordinate dispatch.

For every fixture in ``tests/fixtures/coordinate/<platform>/<action>/NNN/``:

  1. Production :class:`BoundsGenerator` parses the captured XML against
     the captured screenshot to produce the labelled manifest exactly as
     the deployed pipeline would.

  2. Each manifest element is fed through two converters:

       * legacy  — reproduces the pre-fix behaviour (clamping
         device-pixel bounds against the logical screen dimensions).
       * current — the current :class:`CoordinateConverter` honouring
         :class:`CoordinateSystem`.

  3. The current converter must dispatch every element to a logical
     coordinate that falls inside that element's logical bbox. This is
     the strong-property assertion the bug used to violate.

  4. The harness then renders three outputs per case using the
     production primitives (:class:`BoxDrawer` for the manifest;
     :class:`TraceRenderer` for the per-element tap circle, invoked
     once per element on a chained canvas). The two trace images make
     the legacy vs current divergence visually obvious; the assertion
     above is the load-bearing pin.
"""

from __future__ import annotations

import io
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from PIL import Image, ImageDraw

from fathom.constants import ActionType
from fathom.core.artifact.drawing import BoxDrawer
from fathom.core.artifact.renderer import TraceRenderer
from fathom.processing.drawer import BoundsGenerator
from fathom.schemas.actions import Action, Bounds, CoordinateSource, CoordinateSystem
from fathom.schemas.artifact import ArtifactRecord, TracePayload
from fathom.schemas.observation import ElementSource
from fathom.schemas.screens import ScreenCapture
from fathom.utils.coordinates import CoordinateConverter

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "coordinate"
OUTPUT = FIXTURES / "output"


class CoordinateDispatchCaseLoader:
    """
    Walks the fixture tree and yields one descriptor per pinned case.
    """

    def __init__(self, *, root: Path) -> None:
        """
        Bind to the fixture root.
        """

        self.__root = root

    def discover(self, *, platform: str, action: str) -> List[Dict[str, Any]]:
        """
        Return every case under ``<root>/<platform>/<action>/NNN/``.
        """

        action_root = self.__root / platform / action
        if not action_root.is_dir():
            return []

        cases: List[Dict[str, Any]] = []
        for case_dir in sorted(p for p in action_root.iterdir() if p.is_dir()):
            descriptor = self.__load_case(case_dir=case_dir)
            if descriptor:
                cases.append(descriptor)
        return cases

    @staticmethod
    def __load_case(*, case_dir: Path) -> Dict[str, Any]:
        """
        Parse one case directory into a descriptor dict.
        """

        manifest = case_dir / "case.yaml"
        hierarchy = case_dir / "hierarchy.xml"
        screenshot = case_dir / "screenshot.png"
        if not (manifest.is_file() and hierarchy.is_file() and screenshot.is_file()):
            return {}

        with manifest.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

        return {
            "id": case_dir.name,
            "dir": case_dir,
            "case": data,
            "hierarchy": hierarchy,
            "screenshot": screenshot,
        }


class LegacyDispatchSimulator:
    """
    Reproduces the pre-fix ``Bounds.to_pixels`` behaviour exactly.

    The legacy implementation treated ``coordinate_system="pixel"`` as
    pass-through and then clamped device-pixel values against the
    logical screen dimensions. This class preserves that arithmetic so
    the comparison renderings show the historical dispatch coords
    side-by-side with the current ones.
    """

    @staticmethod
    def dispatch_center(
        *,
        bounds: Bounds,
        logical_width: int,
        logical_height: int,
    ) -> Tuple[int, int]:
        """
        Return the dispatched ``(x, y)`` the legacy converter would emit.
        """

        x, y, width, height = bounds.x, bounds.y, bounds.width, bounds.height
        max_x = max(0, logical_width - 1)
        max_y = max(0, logical_height - 1)
        x = max(0, min(x, max_x))
        y = max(0, min(y, max_y))
        width = max(1, min(width, max(1, logical_width - x)))
        height = max(1, min(height, max(1, logical_height - y)))
        return x + width // 2, y + height // 2


class CoordinateDispatchRenderer:
    """
    Render the production-style annotated and trace images for one case.

    Uses :class:`BoxDrawer` for the manifest annotation (identical to
    :class:`PerceptionRenderer`) and invokes :class:`TraceRenderer` once
    per element on a chained canvas to draw the legacy and current
    dispatch overlays. No primitive in this class duplicates production
    drawing logic — the renderers are wired exactly as production wires
    them; only the iteration order is test-specific.
    """

    def __init__(
        self,
        *,
        box_drawer: BoxDrawer = None,
        trace_renderer: TraceRenderer = None,
    ) -> None:
        """
        Bind to production drawing primitives.
        """

        self.__box_drawer = box_drawer or BoxDrawer()
        self.__trace_renderer = trace_renderer or TraceRenderer()

    def render(
        self,
        *,
        case: Dict[str, Any],
        elements: List[Any],
        dispatches: Dict[str, Dict[str, Tuple[int, int]]],
        output_root: Path,
    ) -> None:
        """
        Write ``annotated.png``, ``legacy.png``, and ``current.png`` for the case.
        """

        case_id = case["id"]
        platform = case["case"]["platform"]
        action_type = case["dir"].parent.name

        target_dir = output_root / platform / action_type / case_id
        target_dir.mkdir(parents=True, exist_ok=True)

        annotated = self.__render_annotated(
            screenshot=case["screenshot"],
            elements=elements,
        )
        (target_dir / "annotated.png").write_bytes(annotated)

        for variant in ("legacy", "current"):
            traced = self.__render_traces(
                base=annotated,
                case=case["case"],
                elements=elements,
                dispatches={
                    element.label: dispatches[variant][element.label] for element in elements
                },
            )
            (target_dir / f"{variant}.png").write_bytes(traced)

    def __render_annotated(self, *, screenshot: Path, elements: List[Any]) -> bytes:
        """
        Compose the manifest annotation via the production :class:`BoxDrawer`.
        """

        canvas = Image.open(screenshot).convert("RGB")
        draw = ImageDraw.Draw(canvas, "RGBA")
        for element in elements:
            bounds = element.bounds
            self.__box_drawer.draw(
                canvas=draw,
                bounds=(
                    int(bounds.x1),
                    int(bounds.y1),
                    int(bounds.x2),
                    int(bounds.y2),
                ),
                source=ElementSource.XML,
                label_id=element.label,
                text=(element.attributes or {}).get("text") or None,
            )
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()

    def __render_traces(
        self,
        *,
        base: bytes,
        case: Dict[str, Any],
        elements: List[Any],
        dispatches: Dict[str, Tuple[int, int]],
    ) -> bytes:
        """
        Chain :class:`TraceRenderer` once per element on top of the annotated base.
        """

        scale_x = self.__scale(image_bytes=base) / case["screen"]["logical"]["width"]
        screen = ScreenCapture(
            width=case["screen"]["logical"]["width"],
            height=case["screen"]["logical"]["height"],
            activity="test",
            image=base,
            timestamp=int(time.time() * 1000),
        )

        for element in elements:
            logical_xy = dispatches[element.label]
            pixel_xy = (
                int(logical_xy[0] * scale_x),
                int(logical_xy[1] * scale_x),
            )
            record = ArtifactRecord(
                session_id="test",
                package_name="test",
                step_number=0,
                created=int(time.time() * 1000),
                payload=TracePayload(
                    capture=screen.model_copy(update={"image": screen.image}),
                    coords=pixel_xy,
                    action=Action(
                        action_type=ActionType.TAP,
                        rationale="characterise dispatch",
                        target=element.label,
                        label_id=element.label,
                    ),
                ),
            )
            traced_bytes = self.__trace_renderer.render(record=record)
            screen = screen.model_copy(update={"image": traced_bytes})

        return screen.image

    @staticmethod
    def __scale(*, image_bytes: bytes) -> int:
        """
        Return the pixel width of the encoded PNG.
        """

        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.width


class IosTapDispatchTest(unittest.TestCase):
    """
    Pins coordinate-dispatch correctness for every iOS tap case.
    """

    def test_every_element_dispatches_inside_its_bbox(self) -> None:
        """
        For every manifest element on every fixture screen, the current
        :class:`CoordinateConverter` must dispatch to a logical coord
        inside that element's logical bbox. The pinned case-specific
        coordinate from ``case.yaml`` is also asserted as a sanity pin.
        """

        loader = CoordinateDispatchCaseLoader(root=FIXTURES)
        cases = loader.discover(platform="ios", action="tap")
        self.assertGreater(len(cases), 0, "no fixtures discovered under ios/tap/")

        renderer = CoordinateDispatchRenderer()
        for descriptor in cases:
            with self.subTest(case=descriptor["id"]):
                self.__run_case(descriptor=descriptor, renderer=renderer)

    def __run_case(
        self,
        *,
        descriptor: Dict[str, Any],
        renderer: CoordinateDispatchRenderer,
    ) -> None:
        """
        Execute one fixture: dispatch every element, assert, render.
        """

        case = descriptor["case"]
        logical_width = int(case["screen"]["logical"]["width"])
        logical_height = int(case["screen"]["logical"]["height"])

        with Image.open(descriptor["screenshot"]) as image:
            pixel_width, pixel_height = image.width, image.height

        screenshot_bytes = Path(descriptor["screenshot"]).read_bytes()

        tree = ET.parse(descriptor["hierarchy"])
        elements, _ = BoundsGenerator.create_element(
            root=tree.getroot(),
            image=screenshot_bytes,
            action=ActionType.TAP,
            cv_enabled=False,
        )
        self.assertGreater(
            len(elements),
            0,
            msg=f"case {descriptor['id']!r}: BoundsGenerator produced no elements",
        )

        converter = CoordinateConverter(
            logical_width=logical_width,
            logical_height=logical_height,
            pixel_width=pixel_width,
            pixel_height=pixel_height,
            workflow_id=f"test::{descriptor['id']}",
        )

        dispatches: Dict[str, Dict[str, Tuple[int, int]]] = {
            "current": {},
            "legacy": {},
        }

        scale_x = pixel_width / logical_width
        scale_y = pixel_height / logical_height

        for element in elements:
            bounds = Bounds(
                x=int(element.bounds.x1),
                y=int(element.bounds.y1),
                width=int(element.bounds.x2 - element.bounds.x1),
                height=int(element.bounds.y2 - element.bounds.y1),
                coordinate_system=CoordinateSystem.DEVICE_PIXEL,
                source=CoordinateSource.XML,
            )

            current_xy = converter.center_to_pixels(bounds=bounds)
            legacy_xy = LegacyDispatchSimulator.dispatch_center(
                bounds=bounds,
                logical_width=logical_width,
                logical_height=logical_height,
            )

            dispatches["current"][element.label] = current_xy
            dispatches["legacy"][element.label] = legacy_xy

            logical_x_range = range(
                int(element.bounds.x1 / scale_x),
                int(element.bounds.x2 / scale_x) + 1,
            )
            logical_y_range = range(
                int(element.bounds.y1 / scale_y),
                int(element.bounds.y2 / scale_y) + 1,
            )
            self.assertIn(
                current_xy[0],
                logical_x_range,
                msg=(
                    f"case {descriptor['id']!r}: label_id={element.label!r} current "
                    f"converter dispatched x={current_xy[0]} outside logical bbox "
                    f"x range {logical_x_range.start}..{logical_x_range.stop - 1}."
                ),
            )
            self.assertIn(
                current_xy[1],
                logical_y_range,
                msg=(
                    f"case {descriptor['id']!r}: label_id={element.label!r} current "
                    f"converter dispatched y={current_xy[1]} outside logical bbox "
                    f"y range {logical_y_range.start}..{logical_y_range.stop - 1}."
                ),
            )

        pinned_label = case.get("label_id")
        pinned_dispatch = (case.get("expected") or {}).get("dispatched_logical")
        if pinned_label and pinned_dispatch and pinned_label in dispatches["current"]:
            pinned_expected = (int(pinned_dispatch["x"]), int(pinned_dispatch["y"]))
            self.assertEqual(
                dispatches["current"][pinned_label],
                pinned_expected,
                msg=(
                    f"case {descriptor['id']!r}: pinned label_id={pinned_label!r} "
                    f"dispatch mismatch — got {dispatches['current'][pinned_label]}, "
                    f"case.yaml expected {pinned_expected}."
                ),
            )

        renderer.render(
            case=descriptor,
            elements=elements,
            dispatches=dispatches,
            output_root=OUTPUT,
        )
