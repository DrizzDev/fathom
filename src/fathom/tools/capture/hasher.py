from __future__ import annotations

import hashlib
import io
import xml.etree.ElementTree as ET  # nosec

from PIL import Image

from fathom.schemas.configuration import HasherConfig


class HybridHasher:
    """
    Hybrid screen hashing for efficient deduplication.

    Combines multiple hashing strategies:
    1. Perceptual hash (dHash) - robust to small changes
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

    def __init__(self, config: HasherConfig | None = None) -> None:
        """
        Initialize hasher.

        Args:
            config: Hasher configuration.
        """

        self.__config = config or HasherConfig()

    def compute(self, image: bytes) -> str:
        """
        Compute hybrid hash for image.

        Args:
            image: Image bytes (PNG/JPEG).

        Returns:
            Hexadecimal hash string.
        """

        try:
            img_pil = Image.open(io.BytesIO(image))
            img = img_pil.convert("RGB")

            parts = []

            if self.__config.use_perceptual:
                phash = self.compute_phash(img)
                parts.append(phash)

            if self.__config.use_structural:
                structural_hash = self.compute_structural(img)
                parts.append(structural_hash)

            content_hash = hashlib.md5(image).hexdigest()[:8]  # nosec
            parts.append(content_hash)

            return "-".join(parts)

        except ImportError:
            return hashlib.md5(image).hexdigest()  # nosec

        except Exception:
            return hashlib.md5(image).hexdigest()  # nosec

    def compute_dhash(self, img: Image.Image) -> str:
        """
        Compute difference hash (dHash).

        Much more sensitive to structural changes and gradients than pHash.
        Resizes to (size+1) x size and compares adjacent pixels.
        """

        # dHash with 16x16 = 256 bits (extremely detailed)
        # We need width 17 to compare 16 differences
        hash_size = 16
        img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        gray = img.convert("L")

        pixels = list(gray.getdata())

        diff = []
        for row in range(hash_size):
            for col in range(hash_size):
                # pixel[x] < pixel[x+1]
                left = pixels[row * (hash_size + 1) + col]
                right = pixels[row * (hash_size + 1) + col + 1]
                diff.append(left > right)

        decimal_value = 0
        for index, value in enumerate(diff):
            if value:
                decimal_value += 2**index

        return format(decimal_value, f"0{hash_size * hash_size // 4}x")

    def compute_phash(self, img: Image.Image) -> str:
        """
        Legacy pHash wrapper (redirects to dHash for better performance).
        """
        return self.compute_dhash(img)

    def compute_structural(self, img: Image.Image) -> str:
        """
        Compute structural hash.

        Divides image into grid and computes brightness per cell.

        Args:
            img: PIL Image.

        Returns:
            Hex hash string.
        """

        grid_size = (4, 4)
        thumbnail = img.resize(grid_size, Image.Resampling.LANCZOS)
        gray = thumbnail.convert("L")

        pixels = list(gray.getdata())

        quantized = [p // 32 for p in pixels]
        hex_str = "".join(format(q, "x") for q in quantized)

        return hex_str[:8]

    def similarity(self, hash1: str, hash2: str) -> float:
        """
        Compute similarity between two hashes.

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

        for p1, p2 in zip(parts1, parts2, strict=False):
            sim = self.__hamming_similarity(p1, p2)
            similarities.append(sim)

        return sum(similarities) / len(similarities)

    def __hamming_similarity(self, hex1: str, hex2: str) -> float:
        """
        Compute Hamming similarity between hex strings.

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
        """
        Check if two hashes are similar.

        Args:
            hash1: First hash.
            hash2: Second hash.
            threshold: Similarity threshold.

        Returns:
            True if similarity >= threshold.
        """

        return self.similarity(hash1, hash2) >= threshold


class FastHasher:
    """
    Fast content-only hasher for simple deduplication.
    Uses MD5 for exact matching without image processing. Suitable when PIL is not available.
    """

    def compute(self, image: bytes) -> str:
        """
        Compute MD5 hash of image.

        Args:
            image: Image bytes.

        Returns:
            MD5 hex string.
        """

        return hashlib.md5(image).hexdigest()  # nosec

    def similarity(self, hash1: str, hash2: str) -> float:
        """
        Check exact match.

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
        """
        Check exact match.

        Args:
            hash1: First hash.
            hash2: Second hash.
            threshold: Ignored for exact matching.

        Returns:
            True if hashes are equal.
        """

        return hash1 == hash2


class TreeHasher:
    """
    Robust XML state hasher using structural and interaction analysis.

    Generates two hashes:
    1. Structural Hash: Represents the complete DOM tree structure and content.
    2. Interaction Hash: Represents the set of actionable elements (buttons, inputs).
    """

    def compute(self, xml_content: str) -> tuple[str, str]:
        """
        Compute structural and interaction hashes from XML.

        Returns:
            (structural_hash, interaction_hash)
        """
        if not xml_content:
            return "0" * 16, "0" * 16

        try:
            # Parse XML
            # Use basic string cleaning to handle potential encoding issues
            clean_xml = xml_content.strip()
            # Remove XML declaration if present to avoid parse errors
            if clean_xml.startswith("<?xml"):
                clean_xml = clean_xml.split("?>", 1)[-1]

            root = ET.fromstring(clean_xml)  # nosec

            structural_sig = []
            interaction_sig = []

            # DFS Traversal
            stack = [(root, 0)]

            while stack:
                node, depth = stack.pop()

                # 1. Structural Signature
                # Tag, Class, Resource-ID, Text, Content-Desc, Checked, Selected
                # We ignore bounds/index/focused as they are volatile
                clazz = node.get("class", "")
                res_id = node.get("resource-id", "")
                text = node.get("text", "")
                desc = node.get("content-desc", "")
                checked = node.get("checked", "false")
                selected = node.get("selected", "false")
                enabled = node.get("enabled", "true")

                # Create a dense node signature
                # Format: D{depth}|C{class}|I{id}|T{text}|D{desc}|S{checked}{selected}{enabled}
                node_sig = (
                    f"D{depth}|C{clazz}|I{res_id}|T{text}|D{desc}|S{checked}{selected}{enabled}"
                )
                structural_sig.append(node_sig)

                # 2. Interaction Signature
                # If element is actionable, add it to interaction set
                clickable = node.get("clickable", "false") == "true"
                long_clickable = node.get("long-clickable", "false") == "true"
                check_able = node.get("checkable", "false") == "true"
                is_enabled = enabled == "true"

                if is_enabled and (clickable or long_clickable or check_able):
                    # For interactions, we care about WHAT can be clicked.
                    # We include bounds here because if a button moves significantly,
                    # it might be a new layout state, but we round them to grid to avoid noise.
                    # Actually, let's strictly stick to semantic identity for now to avoid scroll noise.
                    # Just the ID/Text/Desc is usually enough to identify the "set of actions".
                    interaction_sig.append(f"{res_id}|{text}|{desc}")

                # Add children to stack (reverse order to preserve document order in DFS)
                for child in reversed(node):
                    stack.append((child, depth + 1))

            # Compute MD5 hashes
            struct_str = "".join(structural_sig)
            inter_str = "|".join(
                sorted(interaction_sig)
            )  # Sort interactions to be order-independent? Or Keep document order?
            # Document order is better for "Screen Structure", but Sorted is better for "Set of Actions".
            # Let's keep document order for now as UI logic usually implies order matters.

            s_hash = hashlib.md5(struct_str.encode()).hexdigest()[:16]  # nosec
            i_hash = hashlib.md5(inter_str.encode()).hexdigest()[:16]  # nosec

            return s_hash, i_hash

        except Exception:
            # Fallback for invalid XML
            return "0" * 16, "0" * 16
