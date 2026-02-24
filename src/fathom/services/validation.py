"""
Validation service for screen item pre-validation.

Provides:
- Requirement extraction from intent strings
- Screen validation against requirements using vision + hierarchy fallback
- Detailed validation results with confidence scores
"""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as ET  # nosec
from datetime import datetime, timezone
from logging import getLogger
from typing import List, Optional

from fathom.prompts.modes import PromptMode
from fathom.schemas.results import AnalysisResult
from fathom.schemas.screens import ScreenCapture
from fathom.schemas.validation import (
    ValidationItemResult,
    ValidationRequirement,
    ValidationResult,
)
from fathom.services.hierarchy import HierarchyService
from fathom.tools.vision import VisionTool

logger = getLogger(__name__)


class ValidationService:
    """
    Manages screen item validation before and after actions.

    Validates that required UI elements (specified in intent) exist
    on the current screen. Uses vision model + fallback to XML hierarchy
    for fast element detection.

    Thread-safe and injectable into workflows.
    """

    def __init__(
        self,
        vision_tool: Optional[VisionTool] = None,
        hierarchy_service: Optional[HierarchyService] = None,
    ) -> None:
        """
        Initialize validation service.

        Args:
            vision_tool: Vision tool for LLM-based element detection. If None,
                falls back to hierarchy-only validation.
            hierarchy_service: Hierarchy service for fast XML element lookup.
        """
        self._vision_tool = vision_tool
        self._hierarchy_service = hierarchy_service
        self._cache: dict[str, ValidationResult] = {}

    def extract_requirements(self, intent: str) -> List[ValidationRequirement]:
        """
        Parse intent string to extract validation requirements.

        Recognizes patterns like:
        - "requires: element1, element2" (comma-separated)
        - "ensure [element] is visible"
        - "check that [element] exists"
        - "validate: item1; item2"
        - "validate if the price of [item] is..."
        - "verify price of [item]"
        - "check price of [item]"

        Args:
            intent: User intent string

        Returns:
            List of ValidationRequirement objects, empty if no validation clauses found.
        """
        requirements: List[ValidationRequirement] = []

        # Pattern 1: "requires: item1, item2, item3"
        requires_match = re.search(
            r"requires?\s*:\s*([^.!?]+?)(?:\.|$)",
            intent,
            re.IGNORECASE,
        )
        if requires_match:
            items_str = requires_match.group(1)
            items = [item.strip() for item in re.split(r"[,;]", items_str) if item.strip()]
            for item in items:
                requirements.append(
                    ValidationRequirement(
                        item_name=item,
                        description=None,
                        match_type="fuzzy",
                        severity="critical",
                    )
                )

        # Pattern 2: "ensure [element] is visible"
        ensure_matches = re.findall(
            r"ensure\s+([^.!?,]+?)\s+(?:is\s+)?visible",
            intent,
            re.IGNORECASE,
        )
        for item in ensure_matches:
            item = item.strip()
            if item not in [r.item_name for r in requirements]:
                requirements.append(
                    ValidationRequirement(
                        item_name=item,
                        description=None,
                        match_type="fuzzy",
                        severity="critical",
                    )
                )

        # Pattern 3: "check that [element] exists"
        check_matches = re.findall(
            r"check\s+that\s+([^.!?,]+?)\s+exists",
            intent,
            re.IGNORECASE,
        )
        for item in check_matches:
            item = item.strip()
            if item not in [r.item_name for r in requirements]:
                requirements.append(
                    ValidationRequirement(
                        item_name=item,
                        description=None,
                        match_type="fuzzy",
                        severity="critical",
                    )
                )

        # Pattern 4: "validate: item1; item2"
        validate_match = re.search(
            r"validate\s*:\s*([^.!?]+?)(?:\.|$)",
            intent,
            re.IGNORECASE,
        )
        if validate_match:
            items_str = validate_match.group(1)
            items = [item.strip() for item in re.split(r"[,;]", items_str) if item.strip()]
            for item in items:
                if item not in [r.item_name for r in requirements]:
                    requirements.append(
                        ValidationRequirement(
                            item_name=item,
                            description=None,
                            match_type="fuzzy",
                            severity="critical",
                        )
                    )

        # Pattern 5: Price validation patterns
        # Recognizes: "validate if the price of X", "validate price of X", "verify price of X"
        price_patterns = [
            r"validate\s+(?:if\s+the\s+)?price\s+of\s+(\w+(?:\s+\w+)??)(?:\s+(?:is|and|verify|validate|check)|\.|\?|!|,|;|$)",
            r"verify\s+(?:the\s+)?price\s+of\s+(\w+(?:\s+\w+)??)(?:\s+(?:and|verify|validate|check|before)|\.|\?|!|,|;|$)",
            r"check\s+(?:the\s+)?price\s+of\s+(\w+(?:\s+\w+)??)(?:\s+(?:and|verify|validate|check)|\.|\?|!|,|;|$)",
        ]
        for pattern in price_patterns:
            price_matches = re.findall(pattern, intent, re.IGNORECASE)
            for item in price_matches:
                item = item.strip()
                price_item = f"{item} price"
                if price_item not in [r.item_name for r in requirements]:
                    requirements.append(
                        ValidationRequirement(
                            item_name=price_item,
                            description=f"Validate price of {item}",
                            match_type="fuzzy",
                            severity="critical",
                        )
                    )

        logger.info(f"Extracted {len(requirements)} validation requirements from intent")
        return requirements

    async def validate_screen(
        self,
        screen: ScreenCapture,
        requirements: List[ValidationRequirement],
        timeout_seconds: float = 5.0,
    ) -> ValidationResult:
        """
        Validate that required items are present on the given screen.

        Uses vision model (if available) to check for element presence from screenshot.
        Falls back to XML hierarchy parsing for fast element name matching.
        Handles timeouts gracefully.

        Args:
            screen: Screen capture to validate against
            requirements: List of required items to check
            timeout_seconds: Max time to spend on validation

        Returns:
            ValidationResult with detailed pass/fail per item.
        """
        if not requirements:
            return ValidationResult(
                passed=True,
                items=[],
                missing_items=[],
                found_items=[],
                confidence_score=1.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                notes="No validation requirements specified.",
            )

        item_results: List[ValidationItemResult] = []
        found_items: List[str] = []
        missing_items: List[str] = []

        # Try vision-based validation first if available
        if self._vision_tool:
            item_results, found_items, missing_items = await self._validate_with_vision(
                screen, requirements, timeout_seconds
            )
        else:
            logger.debug("Vision tool not available, using hierarchy-based validation")

        # Fallback to hierarchy-based matching if no results yet
        if not item_results and self._hierarchy_service:
            item_results, found_items, missing_items = await self._validate_with_hierarchy(
                screen, requirements, item_results
            )

        # If we still have no results, mark all items as missing (validation failed)
        if not item_results:
            logger.warning("Validation could not check any items (no vision or hierarchy)")
            missing_items = [r.item_name for r in requirements]
            found_items = []

        # Determine pass/fail: all critical items must be found
        critical_items = [r for r in requirements if r.severity == "critical"]
        critical_missing = [
            m for m in missing_items if any(cr.item_name == m for cr in critical_items)
        ]
        passed = len(critical_missing) == 0

        # Calculate confidence: percentage of items found
        total_items = len(requirements)
        confidence_score = len(found_items) / total_items if total_items > 0 else 1.0

        result = ValidationResult(
            passed=passed,
            items=item_results,
            missing_items=missing_items,
            found_items=found_items,
            confidence_score=confidence_score,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes=self._build_validation_summary(
                passed, found_items, missing_items, confidence_score
            ),
        )

        logger.info(f"Validation result: {result.notes}")
        return result

    async def _validate_with_vision(
        self,
        screen: ScreenCapture,
        requirements: List[ValidationRequirement],
        timeout_seconds: float = 5.0,
    ) -> tuple[List[ValidationItemResult], List[str], List[str]]:
        """
        Validate using vision model (LLM) to check element presence.

        Args:
            screen: Screen capture
            requirements: Items to check for
            timeout_seconds: Timeout for vision call (default 5s for validation checks)

        Returns:
            Tuple of (item_results, found_items, missing_items)
        """
        try:
            # Check if vision tool is available
            if not self._vision_tool:
                logger.debug("Vision tool not available for validation")
                return [], [], []
            vision_tool = self._vision_tool

            # Build a validation context for the vision model
            # Check if we're validating prices
            is_price_validation = any("price" in r.item_name.lower() for r in requirements)

            if is_price_validation:
                required_items = ", ".join([r.item_name for r in requirements])
                context = (
                    f"Check if the following UI elements with price information are visible on this screen: "
                    f"{required_items}. For each item, confirm if it is visible and has price info. "
                    f"Respond with YES for visible items, NO for missing items."
                )
            else:
                required_items = ", ".join([r.item_name for r in requirements])
                context = (
                    f"Check if the following UI elements are visible on this screen: "
                    f"{required_items}. For each item, respond with YES if visible or NO if not."
                )

            logger.debug(f"Running vision validation with context: {context}")

            # Use consistent timeout (5 seconds max for validation)
            effective_timeout = timeout_seconds

            async def _run_validation(timeout: float) -> AnalysisResult:
                return await asyncio.wait_for(
                    vision_tool.analyze(
                        intent=context,
                        capture=screen,
                        use_xml=False,
                        mode=PromptMode.VERIFICATION,
                    ),
                    timeout=timeout,
                )

            try:
                result = await _run_validation(effective_timeout)
            except asyncio.TimeoutError:
                retry_timeout = max(effective_timeout, 10.0)
                result = await _run_validation(retry_timeout)

            # Parse vision response
            item_results: List[ValidationItemResult] = []
            found_items: List[str] = []
            missing_items: List[str] = []

            if result and hasattr(result, "reasoning"):
                response_text = str(result.reasoning).lower()
                for req in requirements:
                    item_lower = req.item_name.lower()
                    # Simple heuristic: check if item name appears with YES
                    if f"{item_lower}" in response_text and "yes" in response_text:
                        item_results.append(
                            ValidationItemResult(
                                item_name=req.item_name,
                                found=True,
                                confidence=0.85,
                                details="Vision model detected element",
                            )
                        )
                        found_items.append(req.item_name)
                    else:
                        item_results.append(
                            ValidationItemResult(
                                item_name=req.item_name,
                                found=False,
                                confidence=0.15,
                                details="Vision model did not detect element",
                            )
                        )
                        missing_items.append(req.item_name)

            return item_results, found_items, missing_items

        except asyncio.TimeoutError:
            return [], [], []
        except Exception as e:
            logger.warning(f"Vision validation failed: {e}")
            return [], [], []

    async def _validate_with_hierarchy(
        self,
        screen: ScreenCapture,
        requirements: List[ValidationRequirement],
        existing_results: List[ValidationItemResult],
    ) -> tuple[List[ValidationItemResult], List[str], List[str]]:
        """
        Validate using XML hierarchy parsing (fallback to vision).

        Performs fuzzy matching on XML element text and content-desc attributes.

        Args:
            screen: Screen capture
            requirements: Items to check for
            existing_results: Already-validated items (won't re-validate these)

        Returns:
            Tuple of (item_results, found_items, missing_items)
        """
        try:
            item_results = existing_results.copy()
            found_items = [r.item_name for r in existing_results if r.found]
            missing_items = [r.item_name for r in existing_results if not r.found]

            if not screen.xml_content:
                logger.debug("No XML hierarchy available for hierarchy validation")
                return item_results, found_items, missing_items

            # Parse XML and extract text content
            xml_texts = self._extract_xml_texts(screen.xml_content)
            logger.debug(f"Extracted {len(xml_texts)} text elements from XML")

            # Check each requirement against XML
            for req in requirements:
                # Skip if already validated by vision
                if any(r.item_name == req.item_name for r in existing_results):
                    continue

                # Check if this is a price validation requirement
                is_price_req = "price" in req.item_name.lower()

                # For price requirements, use special matching logic
                if is_price_req:
                    # Extract base item name (e.g., "onion" from "the onion price")
                    base_item = req.item_name.replace(" price", "").lower()
                    # Look for base item and any price indicators nearby
                    found = self._match_price_element(base_item, xml_texts)
                else:
                    # Fuzzy match against XML texts
                    found = self._fuzzy_match_element(req.item_name, xml_texts)

                if found:
                    item_results.append(
                        ValidationItemResult(
                            item_name=req.item_name,
                            found=True,
                            confidence=0.75 if not is_price_req else 0.65,
                            details=f"XML hierarchy match: {found}",
                        )
                    )
                    found_items.append(req.item_name)
                else:
                    item_results.append(
                        ValidationItemResult(
                            item_name=req.item_name,
                            found=False,
                            confidence=0.1,
                            details="No XML hierarchy match",
                        )
                    )
                    missing_items.append(req.item_name)

            return item_results, found_items, missing_items

        except Exception as e:
            logger.warning(f"Hierarchy validation failed: {e}")
            found_from_existing = [r.item_name for r in existing_results if r.found]
            missing_from_existing = [r.item_name for r in existing_results if not r.found]
            return existing_results, found_from_existing, missing_from_existing

    def _extract_xml_texts(self, xml_str: str) -> List[str]:
        """
        Extract all text content from XML hierarchy.

        Args:
            xml_str: XML string to parse

        Returns:
            List of text elements found in XML
        """
        texts: List[str] = []
        try:
            root = ET.fromstring(xml_str)  # nosec - XML from known Android UI source
            # Extract text from all elements and content-desc attributes
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    texts.append(elem.text.strip().lower())
                if "content-desc" in elem.attrib:
                    texts.append(elem.attrib["content-desc"].lower())
                if "text" in elem.attrib:
                    texts.append(elem.attrib["text"].lower())
        except Exception as e:
            logger.debug(f"Failed to parse XML: {e}")

        return texts

    def _fuzzy_match_element(self, item_name: str, available_texts: List[str]) -> Optional[str]:
        """
        Perform fuzzy matching of required item against available text.

        Uses simple substring matching with case-insensitivity.
        Can be extended with Levenshtein distance for more sophisticated matching.

        Args:
            item_name: Name of item to find
            available_texts: List of available text from screen

        Returns:
            Matched text if found, None otherwise
        """
        item_lower = item_name.lower()

        # Exact match (highest confidence)
        if item_lower in available_texts:
            return item_lower

        # Substring match
        for text in available_texts:
            if item_lower in text or text in item_lower:
                return text

        # No match found
        return None

    def _match_price_element(self, base_item: str, available_texts: List[str]) -> Optional[str]:
        """
        Match price elements by looking for base item and price indicators.

        For price validation, looks for the base item (e.g., "onion") and
        checks if any price-like indicators (currency, numbers) are nearby.

        Args:
            base_item: Base item name without "price" suffix
            available_texts: List of available text from screen

        Returns:
            Matched text if found, None otherwise
        """
        base_item_lower = base_item.lower().strip()
        price_pattern = r"[\$₹€¥£]|^\d+\.?\d*$"  # Currency symbols or numbers

        # Look for base item in available texts
        for text in available_texts:
            text_lower = text.lower()
            # Check if base item appears in text
            if base_item_lower in text_lower:
                # Found the item, now check if there's any price indicator nearby
                # If the text contains numbers or currency, it's likely a price
                if re.search(price_pattern, text):
                    return text
                # Even without price indicator, finding the item is enough
                return text

        # Try to find any price indicator if item wasn't found exactly
        # (e.g., item might be in a label, price in another field)
        for text in available_texts:
            if re.search(price_pattern, text):
                # Found a price indicator, that's good enough for price validation
                return text
        return None

    def _build_validation_summary(
        self,
        passed: bool,
        found_items: List[str],
        missing_items: List[str],
        confidence: float,
    ) -> str:
        """
        Build human-readable validation summary for logging.

        Args:
            passed: Whether validation passed
            found_items: Items that were found
            missing_items: Items that were not found
            confidence: Confidence score

        Returns:
            Summary string
        """
        status = "PASSED" if passed else "FAILED"
        summary = f"Validation {status}. Found {len(found_items)} items. "

        if missing_items:
            summary += f"Missing: {', '.join(missing_items)}. "

        summary += f"Confidence: {confidence:.1%}"

        return summary
