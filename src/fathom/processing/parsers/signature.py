from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ElementTree  # nosec
from typing import List

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.constants.screen import (
    BOUNDS_DIGEST_LENGTH,
    SIGNATURE_TEXT_PREVIEW_LENGTH,
    SIGNATURE_VALUE_PREVIEW_LENGTH,
)
from fathom.processing.parsers.base import PlatformParser
from fathom.processing.parsers.factory import PlatformParserFactory
from fathom.schemas.hierarchy import NormalizedHierarchyNodeSignature


class HierarchySignatureBuilder:
    """
    Build normalized structural signatures from platform-specific hierarchy XML.
    """

    def compute_hash(self, *, xml_content: str) -> str:
        """
        Compute a stable structural hash for the provided hierarchy XML.
        """

        normalized_xml = self.__trim_xml(xml_content=xml_content)
        root = ElementTree.fromstring(normalized_xml)  # nosec
        parser = PlatformParserFactory.get_parser(root=root)

        signature = self.__build_tree_signature(
            depth=0,
            node=root,
            parser=parser,
        )
        return hashlib.md5(signature.encode("utf-8"), usedforsecurity=False).hexdigest()[
            :VISUAL_HASH_LENGTH
        ]

    def __trim_xml(self, *, xml_content: str) -> str:
        """
        Trim any non-XML wrapper bytes around the hierarchy payload.
        """

        start_index = xml_content.find("<")
        end_index = xml_content.rfind(">")

        if start_index != -1 and end_index != -1:
            return xml_content[start_index : end_index + 1]

        return xml_content

    def __build_tree_signature(
        self,
        *,
        depth: int,
        parser: PlatformParser,
        node: ElementTree.Element,
    ) -> str:
        """
        Recursively build a structural signature for one hierarchy node.
        """

        metadata = parser.build_signature_metadata(node=node)

        if parser.should_ignore_signature_node(metadata=metadata):
            return ""

        signature = (
            f"{depth}:{metadata.class_name}#{metadata.identifier}"
            f"[{metadata.text[:SIGNATURE_TEXT_PREVIEW_LENGTH]}|"
            f"{metadata.content_description[:SIGNATURE_TEXT_PREVIEW_LENGTH]}]"
            f"{self.__build_state_flags(metadata=metadata)}"
            f"{self.__build_value_suffix(metadata=metadata)}"
        )

        child_signatures: List[str] = []

        for child_node in node:
            child_signature = self.__build_tree_signature(
                parser=parser,
                node=child_node,
                depth=depth + 1,
            )
            if child_signature:
                child_signatures.append(child_signature)

        bounds_suffix = self.__build_scroll_bounds_suffix(
            node=node,
            parser=parser,
            metadata=metadata,
        )
        if bounds_suffix:
            signature = f"{signature}@{bounds_suffix}"

        if child_signatures:
            return f"({signature}[" + ",".join(child_signatures) + "])"

        return signature

    def __build_state_flags(self, *, metadata: NormalizedHierarchyNodeSignature) -> str:
        """
        Encode boolean node state into a compact suffix.
        """

        state_flags = ""

        if metadata.is_checked:
            state_flags += "C"

        if metadata.is_selected:
            state_flags += "S"

        if metadata.is_focused:
            state_flags += "F"

        return state_flags

    def __build_value_suffix(
        self,
        *,
        metadata: NormalizedHierarchyNodeSignature,
    ) -> str:
        """
        Include small value payloads only for value-sensitive controls.
        """

        if not metadata.include_value_in_signature:
            return ""

        return f"~{metadata.raw_value[:SIGNATURE_VALUE_PREVIEW_LENGTH]}"

    def __build_scroll_bounds_suffix(
        self,
        *,
        parser: PlatformParser,
        node: ElementTree.Element,
        metadata: NormalizedHierarchyNodeSignature,
    ) -> str:
        """
        Hash direct-child bounds for scrollable containers.
        """

        if not metadata.is_scrollable:
            return ""

        bounds_parts: List[str] = []

        for child_node in node:
            child_metadata = parser.build_signature_metadata(node=child_node)
            if child_metadata.bounds:
                bounds_parts.append(child_metadata.bounds)

        if not bounds_parts:
            return ""

        return hashlib.md5(
            ",".join(bounds_parts).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:BOUNDS_DIGEST_LENGTH]
