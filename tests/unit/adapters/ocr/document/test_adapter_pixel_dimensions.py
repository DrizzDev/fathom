from __future__ import annotations

import io
import unittest
from typing import Any, List, Tuple

from PIL import Image

from fathom.adapters.ocr.document.adapter import DocumentAiOcr
from fathom.schemas.budgets import PerceptionBudget
from fathom.schemas.ocr import DocumentAiConfiguration
from fathom.schemas.screens import ScreenCapture


def _png(*, width: int, height: int) -> bytes:
    """
    Encode a blank PNG at an explicit pixel size.
    """

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class _RecordingMapper:
    """
    Stands in for :class:`DocumentAiMapper` and records the dimensions it is
    handed, which is the whole contract under test.
    """

    def __init__(self) -> None:
        self.calls: List[Tuple[int, int]] = []

    def map_document(self, *, document: Any, width: int, height: int) -> Tuple[Any, ...]:
        _ = document
        self.calls.append((width, height))
        return ()


class _StubDocument:
    """Opaque stand-in for a Document AI ``Document``."""


class _StubResponse:
    """Carries ``.document``, which is all the adapter reads off the RPC."""

    document = _StubDocument()


class _StubClient:
    """
    Minimal Document AI client stub. ``extract`` swallows exceptions and
    degrades to an empty result, so a stub that raises would make these tests
    silently vacuous — it must return a well-formed response.
    """

    @staticmethod
    def process_document(request: Any) -> Any:
        _ = request
        return _StubResponse()


class DocumentAiPixelDimensionsTest(unittest.IsolatedAsyncioTestCase):
    """
    Document AI reports ``normalized_vertices`` in 0..1 of the IMAGE it was
    given, and :class:`DocumentAiMapper` multiplies those by the width/height
    passed in before stamping the result ``DEVICE_PIXEL``. So the dimensions
    handed to the mapper must be the image's own pixel size.

    ``ScreenCapture.width``/``height`` are the LOGICAL point dimensions. Passing
    those produced bounds already in logical space but labelled ``DEVICE_PIXEL``,
    which ``to_logical_dispatch`` then divided by the scale a second time — every
    OCR-resolved tap landed at 1/scale of its correct distance from the origin.

    Live incident 2026-08-06 (iPad Air 4, logical 1180x820 / pixel 2360x1640):
    OCR located "Add Visit" at (936, 631), the tap went to (485, 318) — ~590px
    away. Identical in runs HXPBk, HawUV and QgzIM.
    """

    @staticmethod
    def __adapter(mapper: _RecordingMapper) -> DocumentAiOcr:
        return DocumentAiOcr(
            configuration=DocumentAiConfiguration(
                project="vision-478905",
                location="us",
                processor="proc-1",
            ),
            mapper=mapper,  # type: ignore[arg-type]
            client=_StubClient(),
        )

    @staticmethod
    def __capture(*, logical: Tuple[int, int], pixel: Tuple[int, int]) -> ScreenCapture:
        return ScreenCapture(
            width=logical[0],
            height=logical[1],
            activity="com.example.app",
            image=_png(width=pixel[0], height=pixel[1]),
            timestamp=0,
        )

    @staticmethod
    def __budget() -> PerceptionBudget:
        return PerceptionBudget(ocr=5000, local=5000, localization=5000)

    async def test_maps_against_pixel_size_not_logical_on_retina(self) -> None:
        # The regression: 2x device, so logical and pixel differ by exactly 2.
        mapper = _RecordingMapper()
        await self.__adapter(mapper).extract(
            capture=self.__capture(logical=(1180, 820), pixel=(2360, 1640)),
            budget=self.__budget(),
        )

        self.assertEqual(
            [(2360, 1640)],
            mapper.calls,
            "mapper must receive the PNG's pixel size; passing the logical "
            "1180x820 is what halved every OCR-resolved tap",
        )

    async def test_maps_against_pixel_size_at_three_times_scale(self) -> None:
        # 3x devices skew by a third rather than a half — same defect, worse.
        mapper = _RecordingMapper()
        await self.__adapter(mapper).extract(
            capture=self.__capture(logical=(430, 932), pixel=(1290, 2796)),
            budget=self.__budget(),
        )

        self.assertEqual([(1290, 2796)], mapper.calls)

    async def test_logical_and_pixel_agree_on_non_retina(self) -> None:
        # 1x device: the two spaces coincide, so the old code was accidentally
        # right here. Pinned so a "fix" that only special-cases retina fails.
        mapper = _RecordingMapper()
        await self.__adapter(mapper).extract(
            capture=self.__capture(logical=(800, 600), pixel=(800, 600)),
            budget=self.__budget(),
        )

        self.assertEqual([(800, 600)], mapper.calls)

    async def test_falls_back_to_logical_when_image_is_undecodable(self) -> None:
        # Garbage bytes must not take the OCR pass down — degrade to the logical
        # pair, which is what the adapter used before this fix.
        mapper = _RecordingMapper()
        capture = ScreenCapture(
            width=1180,
            height=820,
            activity="com.example.app",
            image=b"not-a-png",
            timestamp=0,
        )

        await self.__adapter(mapper).extract(capture=capture, budget=self.__budget())

        self.assertEqual([(1180, 820)], mapper.calls)


if __name__ == "__main__":
    unittest.main()
