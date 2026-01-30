"""Hybrid screen hasher for efficient screen comparison."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Optional

from fathom.schemas.configuration import HasherConfig

if TYPE_CHECKING:
    from PIL import Image


class HybridHasher:
    """Hybrid screen hashing for efficient deduplication.

    Combines multiple hashing strategies:
    1. Perceptual hash (pHash) - robust to small changes
    2. Structural hash - based on image structure
    3. Content hash (MD5) - exact matching

    Example:
        ```python
        hasher = HybridHasher()
        hash1 = hasher.compute(screenshot1)
        hash2 = hasher.compute(screenshot2)
        similarity = hasher.similarity(hash1, hash2)
        ```
    """

    def __init__(self, config: Optional[HasherConfig] = None) -> None:
        """Initialize hasher.

        Args:
            config: Hasher configuration.
        """
        self.__config = config or HasherConfig()

    def compute(self, image: bytes) -> str:
        """Compute hybrid hash for image.

        Args:
            image: Image bytes (PNG/JPEG).

        Returns:
            Hexadecimal hash string.
        """
        try:
            import io

            from PIL import Image

            img_pil = Image.open(io.BytesIO(image))
            img = img_pil.convert("RGB")

            parts = []

            if self.__config.use_perceptual:
                phash = self.compute_phash(img)
                parts.append(phash)

            if self.__config.use_structural:
                shash = self.compute_structural(img)
                parts.append(shash)

            content_hash = hashlib.md5(image).hexdigest()[:8]  # nosec
            parts.append(content_hash)

            return "-".join(parts)

        except ImportError:
            return hashlib.md5(image).hexdigest()  # nosec

        except Exception:
            return hashlib.md5(image).hexdigest()  # nosec

    def compute_phash(self, img: "Image.Image") -> str:
        """Compute perceptual hash.

        Uses DCT-based perceptual hashing:
        1. Resize to small thumbnail
        2. Convert to grayscale
        3. Compute mean of pixels
        4. Build binary hash from comparison to mean

        Args:
            img: PIL Image.

        Returns:
            Hex hash string.
        """
        from PIL import Image

        size = self.__config.thumbnail_size
        thumbnail = img.resize(size, Image.Resampling.LANCZOS)
        gray = thumbnail.convert("L")

        pixels = list(gray.getdata())
        mean = sum(pixels) / len(pixels)

        bits = "".join("1" if p > mean else "0" for p in pixels)

        hash_int = int(bits, 2)
        return format(hash_int, "016x")

    def compute_structural(self, img: "Image.Image") -> str:
        """Compute structural hash.

        Divides image into grid and computes brightness per cell.

        Args:
            img: PIL Image.

        Returns:
            Hex hash string.
        """
        from PIL import Image

        grid_size = (4, 4)
        thumbnail = img.resize(grid_size, Image.Resampling.LANCZOS)
        gray = thumbnail.convert("L")

        pixels = list(gray.getdata())

        quantized = [p // 32 for p in pixels]
        hex_str = "".join(format(q, "x") for q in quantized)

        return hex_str[:8]

    def similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two hashes.

        Args:
            hash1: First hash.
            hash2: Second hash.

        Returns:
            Similarity score 0.0 to 1.0.
        """
        if hash1 == hash2:
            return 1.0

        parts1 = hash1.split("-")
        parts2 = hash2.split("-")

        if len(parts1) != len(parts2):
            return 0.0 if hash1 != hash2 else 1.0

        similarities = []

        for p1, p2 in zip(parts1, parts2):
            sim = self.__hamming_similarity(p1, p2)
            similarities.append(sim)

        return sum(similarities) / len(similarities)

    def __hamming_similarity(self, hex1: str, hex2: str) -> float:
        """Compute Hamming similarity between hex strings.

        Args:
            hex1: First hex string.
            hex2: Second hex string.

        Returns:
            Similarity 0.0 to 1.0.
        """
        if len(hex1) != len(hex2):
            return 0.0

        try:
            int1 = int(hex1, 16)
            int2 = int(hex2, 16)
        except ValueError:
            return 0.0 if hex1 != hex2 else 1.0

        xor = int1 ^ int2
        bits_different = bin(xor).count("1")

        max_bits = len(hex1) * 4
        similarity = 1.0 - (bits_different / max_bits)
        return similarity

    def is_similar(
        self,
        hash1: str,
        hash2: str,
        threshold: float = 0.85,
    ) -> bool:
        """Check if two hashes are similar.

        Args:
            hash1: First hash.
            hash2: Second hash.
            threshold: Similarity threshold.

        Returns:
            True if similarity >= threshold.
        """
        return self.similarity(hash1, hash2) >= threshold


class FastHasher:
    """Fast content-only hasher for simple deduplication.

    Uses MD5 for exact matching without image processing.
    Suitable when PIL is not available.
    """

    def compute(self, image: bytes) -> str:
        """Compute MD5 hash of image.

        Args:
            image: Image bytes.

        Returns:
            MD5 hex string.
        """
        return hashlib.md5(image).hexdigest()  # nosec

    def similarity(self, hash1: str, hash2: str) -> float:
        """Check exact match.

        Args:
            hash1: First hash.
            hash2: Second hash.

        Returns:
            1.0 if equal, 0.0 otherwise.
        """
        return 1.0 if hash1 == hash2 else 0.0

    def is_similar(
        self,
        hash1: str,
        hash2: str,
        threshold: float = 0.85,
    ) -> bool:
        """Check exact match.

        Args:
            hash1: First hash.
            hash2: Second hash.
            threshold: Ignored for exact matching.

        Returns:
            True if hashes are equal.
        """
        return hash1 == hash2
