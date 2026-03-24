"""
Unit tests for the validation service and schema models.

Tests cover:
- Requirement extraction from intent strings
- Screen validation logic (vision + hierarchy fallback)
- ValidationResult creation and analysis
- Integration with AgentState
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from fathom.schemas.screens import ScreenCapture
from fathom.schemas.validation import (
    ValidationContext,
    ValidationItemResult,
    ValidationRequirement,
    ValidationResult,
)
from fathom.services.validation import ValidationService


class TestValidationRequirement:
    """Test ValidationRequirement schema."""

    def test_create_critical_requirement(self):
        """Test creating a critical validation requirement."""
        req = ValidationRequirement(
            item_name="login button",
            description="The main login button",
            severity="critical",
        )
        assert req.item_name == "login button"
        assert req.description == "The main login button"
        assert req.severity == "critical"
        assert req.optional is False
        assert req.match_type == "fuzzy"

    def test_create_optional_requirement(self):
        """Test creating an optional validation requirement."""
        req = ValidationRequirement(
            item_name="update banner",
            severity="advisory",
            optional=True,
        )
        assert req.optional is True
        assert req.severity == "advisory"

    def test_requirement_defaults(self):
        """Test that defaults are set correctly."""
        req = ValidationRequirement(item_name="button")
        assert req.match_type == "fuzzy"
        assert req.severity == "critical"
        assert req.optional is False
        assert req.description is None


class TestValidationItemResult:
    """Test ValidationItemResult dataclass."""

    def test_item_found(self):
        """Test result for found item."""
        result = ValidationItemResult(
            item_name="login button",
            found=True,
            confidence=0.95,
            details="Found in XML hierarchy",
        )
        assert result.found is True
        assert result.confidence == 0.95
        assert result.item_name == "login button"

    def test_item_not_found(self):
        """Test result for missing item."""
        result = ValidationItemResult(
            item_name="missing element",
            found=False,
            confidence=0.05,
        )
        assert result.found is False
        assert result.confidence == 0.05


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_passed_validation(self):
        """Test successful validation result."""
        items = [
            ValidationItemResult("button1", True, 0.9),
            ValidationItemResult("button2", True, 0.85),
        ]
        result = ValidationResult(
            passed=True,
            items=items,
            missing_items=[],
            found_items=["button1", "button2"],
            confidence_score=0.875,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes="All items found successfully",
        )
        assert result.passed is True
        assert len(result.items) == 2
        assert len(result.missing_items) == 0
        assert result.confidence_score == 0.875

    def test_failed_validation(self):
        """Test failed validation result."""
        items = [
            ValidationItemResult("button1", True, 0.9),
            ValidationItemResult("button2", False, 0.1),
        ]
        result = ValidationResult(
            passed=False,
            items=items,
            missing_items=["button2"],
            found_items=["button1"],
            confidence_score=0.5,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes="Validation FAILED. Found 1 items. Missing: button2. Confidence: 50.0%",
        )
        assert result.passed is False
        assert len(result.missing_items) == 1
        assert "button2" in result.missing_items


class TestValidationContext:
    """Test ValidationContext helper class."""

    def test_create_context(self):
        """Test creating validation context."""
        requirements = [
            ValidationRequirement(item_name="button1"),
            ValidationRequirement(item_name="button2"),
        ]
        ctx = ValidationContext(requirements, max_retries=5)
        assert len(ctx.requirements) == 2
        assert ctx.retry_count == 0
        assert ctx.should_continue_validating() is True

    def test_retry_counting(self):
        """Test retry counter logic."""
        requirements = [ValidationRequirement(item_name="button")]
        ctx = ValidationContext(requirements, max_retries=3)

        assert ctx.should_continue_validating() is True
        ctx.increment_retry()
        assert ctx.retry_count == 1
        assert ctx.should_continue_validating() is True

        ctx.increment_retry()
        ctx.increment_retry()
        assert ctx.retry_count == 3
        assert ctx.should_continue_validating() is False

    def test_satisfaction_check(self):
        """Test satisfaction checking."""
        requirements = [ValidationRequirement(item_name="button")]
        ctx = ValidationContext(requirements)

        # Not satisfied until result is set
        assert ctx.is_satisfied() is False

        # Set a passing result
        result = ValidationResult(
            passed=True,
            items=[],
            missing_items=[],
            found_items=["button"],
            confidence_score=1.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes="All items found",
        )
        ctx.last_result = result
        assert ctx.is_satisfied() is True

        # Failing result
        failing_result = ValidationResult(
            passed=False,
            items=[],
            missing_items=["button"],
            found_items=[],
            confidence_score=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            notes="Items missing",
        )
        ctx.last_result = failing_result
        assert ctx.is_satisfied() is False


class TestValidationServiceExtraction:
    """Test requirement extraction from intent strings."""

    def test_extract_requires_pattern(self):
        """Test 'requires:' pattern extraction."""
        service = ValidationService()
        intent = "Go to the store, requires: email field, password field, login button."

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 3
        assert "email field" in [r.item_name for r in requirements]
        assert "password field" in [r.item_name for r in requirements]
        assert "login button" in [r.item_name for r in requirements]

    def test_extract_ensure_pattern(self):
        """Test 'ensure ... is visible' pattern."""
        service = ValidationService()
        intent = (
            "Execute the task. Ensure the confirm button is visible and "
            "ensure the error message is visible."
        )

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 2
        assert any("confirm button" in r.item_name for r in requirements)
        assert any("error message" in r.item_name for r in requirements)

    def test_extract_check_pattern(self):
        """Test 'check that ... exists' pattern."""
        service = ValidationService()
        intent = (
            "Complete payment. Check that the total amount exists. "
            "Check that payment button exists."
        )

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 2

    def test_extract_validate_pattern(self):
        """Test 'validate:' pattern."""
        service = ValidationService()
        intent = "Proceed with action. Validate: item1; item2; item3"

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 3

    def test_no_requirements(self):
        """Test intent with no validation requirement patterns."""
        service = ValidationService()
        intent = "Simply tap the button and wait."

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 0

    def test_deduplication(self):
        """Test that duplicate requirements are not created."""
        service = ValidationService()
        intent = "requires: button. Ensure button is visible."

        requirements = service.extract_requirements(intent)
        button_reqs = [r for r in requirements if "button" in r.item_name.lower()]
        # Should have only 1, not 2
        assert len(button_reqs) == 1

    def test_extract_price_validation_pattern_1(self):
        """Test 'validate if the price of [item]' pattern."""
        service = ValidationService()
        intent = (
            "Go to cart and validate if the price of the onion is same "
            "as before and end the session."
        )

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 1
        assert any("onion price" in r.item_name for r in requirements)
        assert any(r.severity == "critical" for r in requirements)

    def test_extract_price_validation_pattern_2(self):
        """Test 'validate price of [item]' pattern."""
        service = ValidationService()
        intent = "Check the bill and validate price of milk and validate price of eggs."

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 2
        assert any("milk price" in r.item_name for r in requirements)
        assert any("eggs price" in r.item_name for r in requirements)

    def test_extract_price_validation_pattern_3(self):
        """Test 'verify price of [item]' pattern."""
        service = ValidationService()
        intent = "Verify price of coffee and verify price of tea before checkout."

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 2
        assert any("coffee price" in r.item_name for r in requirements)
        assert any("tea price" in r.item_name for r in requirements)

    def test_extract_price_validation_pattern_4(self):
        """Test 'check price of [item]' pattern."""
        service = ValidationService()
        intent = "Complete order. Check price of chicken and check price of rice."

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 2


class TestValidationServiceLogic:
    """Test validation logic with mock services."""

    @pytest.mark.asyncio
    async def test_validate_no_requirements(self):
        """Test validation with no requirements (auto-pass)."""
        service = ValidationService()
        screen = MagicMock(spec=ScreenCapture)

        result = await service.validate_screen(screen, [])
        assert result.passed is True
        assert result.confidence_score == 1.0
        assert len(result.items) == 0

    @pytest.mark.asyncio
    async def test_validate_with_hierarchy_fallback(self):
        """Test validation using XML hierarchy fallback."""
        # Create mock hierarchy service
        mock_hierarchy = MagicMock()

        service = ValidationService(
            vision_tool=None,  # No vision, will use hierarchy
            hierarchy_service=mock_hierarchy,
        )

        # Create mock screen
        screen = MagicMock(spec=ScreenCapture)
        screen.xml_content = "<hierarchy><Button text='Login Button'/></hierarchy>"

        requirements = [ValidationRequirement(item_name="Login Button")]

        result = await service.validate_screen(screen, requirements)
        # Should pass because XML contains "Login Button"
        assert result.passed is True
        assert "Login Button" in result.found_items

    @pytest.mark.asyncio
    async def test_validate_missing_items(self):
        """Test validation with missing critical items."""
        service = ValidationService()
        screen = MagicMock(spec=ScreenCapture)
        screen.xml_content = "<hierarchy></hierarchy>"  # Empty XML

        requirements = [
            ValidationRequirement(item_name="button1", severity="critical"),
            ValidationRequirement(item_name="button2", severity="critical"),
        ]

        result = await service.validate_screen(screen, requirements)
        assert result.passed is False
        assert len(result.missing_items) == 2

    @pytest.mark.asyncio
    async def test_validate_mixed_requirements(self):
        """Test validation with both critical and advisory items."""
        # Create mock hierarchy service
        mock_hierarchy = MagicMock()

        service = ValidationService(
            vision_tool=None,
            hierarchy_service=mock_hierarchy,
        )
        screen = MagicMock(spec=ScreenCapture)
        screen.xml_content = "<hierarchy><Button text='critical button'/></hierarchy>"

        requirements = [
            ValidationRequirement(
                item_name="critical button",
                severity="critical",
            ),
            ValidationRequirement(
                item_name="advisory element",
                severity="advisory",
            ),
        ]

        result = await service.validate_screen(screen, requirements)
        # Should pass because all critical items are present (advisory items are optional)
        assert result.passed is True
        assert "critical button" in result.found_items


class TestValidationServiceHelpers:
    """Test helper methods in ValidationService."""

    def test_extract_xml_texts(self):
        """Test XML text extraction."""
        service = ValidationService()
        xml = """
        <hierarchy>
            <Button text="Login" content-desc="login button"/>
            <TextView text="Welcome"/>
        </hierarchy>
        """

        texts = service._extract_xml_texts(xml)
        assert "login" in texts
        assert "welcome" in texts
        assert "login button" in texts

    def test_fuzzy_match_exact(self):
        """Test exact match in fuzzy matching."""
        service = ValidationService()
        available = ["login button", "signup button", "logout button"]

        result = service._fuzzy_match_element("login button", available)
        assert result == "login button"

    def test_fuzzy_match_substring(self):
        """Test substring match."""
        service = ValidationService()
        available = ["main login button", "secondary button"]

        result = service._fuzzy_match_element("login button", available)
        assert result == "main login button"

    def test_fuzzy_match_not_found(self):
        """Test fuzzy match with no match."""
        service = ValidationService()
        available = ["button1", "button2"]

        result = service._fuzzy_match_element("missing", available)
        assert result is None

    def test_build_validation_summary(self):
        """Test validation summary building."""
        service = ValidationService()
        summary = service._build_validation_summary(
            passed=False,
            found_items=["button1"],
            missing_items=["button2", "button3"],
            confidence=0.33,
        )

        assert "FAILED" in summary
        assert "button2" in summary
        assert "button3" in summary
        assert "33.0%" in summary


# Integration-style tests
class TestValidationIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_full_validation_flow(self):
        """Test the complete validation flow."""
        mock_hierarchy = MagicMock()
        service = ValidationService(vision_tool=None, hierarchy_service=mock_hierarchy)

        # Simulate intent with validation requirements
        intent = (
            "Go to the app. Requires: login button, email field, password field. Then tap login."
        )

        requirements = service.extract_requirements(intent)
        assert len(requirements) == 3

        # Create mock screen
        screen = MagicMock(spec=ScreenCapture)
        screen.xml_content = """
        <hierarchy>
            <EditText content-desc="Email field"/>
            <EditText content-desc="Password field"/>
            <Button text="Login Button"/>
        </hierarchy>
        """

        result = await service.validate_screen(screen, requirements)
        assert result.passed is True
        assert result.confidence_score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
