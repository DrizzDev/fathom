from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET

from fathom.app_graph.models import Element, Locator, Screen


class Parser:
    """Translates observed native hierarchy attributes into locator knowledge."""

    def enrich(self, *, screen: Screen, xml: str | None, package: str) -> Screen:
        """Preserve native selectors and avoid inventing missing attributes."""
        if not xml:
            return screen
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            return screen
        elements: list[Element] = []
        structural: set[str] = set()
        selected: set[str] = set()
        for node in root.iter():
            attributes = node.attrib
            if attributes.get("package") not in (None, "", package):
                continue
            resource = attributes.get("resource-id", "")
            description = attributes.get("content-desc", "")
            text = attributes.get("text", "")
            locators: list[Locator] = []
            if resource:
                locators.append(Locator(strategy="resource_id", value=resource))
                if resource.startswith(package + ":id/"):
                    structural.add(resource)
            if description:
                locators.append(Locator(strategy="content_desc", value=description))
            if text:
                locators.append(Locator(strategy="text", value=text))
            if attributes.get("selected") == "true":
                selected.add(description or text or resource)
            if locators:
                elements.append(
                    Element(
                        name=description or text or resource,
                        locators=locators,
                        interactive=attributes.get("clickable") == "true"
                        or attributes.get("focusable") == "true",
                        element_type=attributes.get("class", node.tag),
                        bounds=attributes.get("bounds"),
                    )
                )
        signature = None
        if len(structural) > 3:
            signature = hashlib.sha256(
                json.dumps([sorted(structural), sorted(selected)]).encode()
            ).hexdigest()
        return screen.model_copy(update={"elements": elements, "structure_hash": signature})
