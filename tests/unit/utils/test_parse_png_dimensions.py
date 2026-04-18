"""Tests for fathom.utils.image.parse_png_dimensions."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from fathom.utils.image import parse_png_dimensions


def _make_minimal_png(width: int, height: int) -> bytes:
    """Build a minimal valid PNG with the given IHDR dimensions."""
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(0, 0, 0))
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestParsePngDimensions:
    def test_portrait_dimensions(self) -> None:
        data = _make_minimal_png(1344, 2992)
        w, h = parse_png_dimensions(data)
        assert (w, h) == (1344, 2992)

    def test_landscape_dimensions(self) -> None:
        data = _make_minimal_png(2992, 1344)
        w, h = parse_png_dimensions(data)
        assert (w, h) == (2992, 1344)

    def test_square_dimensions(self) -> None:
        data = _make_minimal_png(1080, 1080)
        w, h = parse_png_dimensions(data)
        assert (w, h) == (1080, 1080)

    def test_small_image(self) -> None:
        data = _make_minimal_png(1, 1)
        w, h = parse_png_dimensions(data)
        assert (w, h) == (1, 1)

    def test_truncated_bytes_raises(self) -> None:
        data = _make_minimal_png(100, 200)
        with pytest.raises(ValueError, match="too small"):
            parse_png_dimensions(data[:20])

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            parse_png_dimensions(b"")

    def test_non_png_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="missing PNG signature"):
            parse_png_dimensions(b"\xff\xd8\xff\xe0" + b"\x00" * 24)

    def test_corrupted_ihdr_chunk_type_raises(self) -> None:
        data = bytearray(_make_minimal_png(100, 200))
        data[12:16] = b"XXXX"
        with pytest.raises(ValueError, match="IHDR chunk not found"):
            parse_png_dimensions(bytes(data))

    def test_matches_pillow_decode(self) -> None:
        """parse_png_dimensions must agree with PIL for various sizes."""
        for w, h in [(1080, 1920), (1920, 1080), (2992, 1344), (1344, 2992)]:
            data = _make_minimal_png(w, h)
            parsed = parse_png_dimensions(data)
            with Image.open(io.BytesIO(data)) as img:
                assert parsed == img.size
