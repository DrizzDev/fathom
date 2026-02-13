import os
from unittest.mock import AsyncMock

import pytest

from fathom.constants import ActionType
from fathom.schemas.actions import Action
from fathom.services.resolution import ReferenceResolutionService


@pytest.mark.asyncio
async def test_resolve_memory_reference():
    # Setup
    mock_ledger = AsyncMock()
    mock_ledger.get.return_value = "secret_value"

    service = ReferenceResolutionService(ledger=mock_ledger)

    action = Action(
        action_type=ActionType.TYPE,
        text="Type $memory.password here",
        rationale="Testing memory resolution",
        target="input_field",
    )

    # Execute
    resolved = await service.resolve(action)

    # Verify
    assert resolved.text == "Type secret_value here"
    mock_ledger.get.assert_called_with("password")


@pytest.mark.asyncio
async def test_resolve_env_reference():
    # Setup
    mock_ledger = AsyncMock()
    os.environ["TEST_API_KEY"] = "12345"

    service = ReferenceResolutionService(ledger=mock_ledger)

    action = Action(
        action_type=ActionType.TYPE,
        text="Key is $env.TEST_API_KEY",
        rationale="Testing env resolution",
        target="input_field",
    )

    # Execute
    resolved = await service.resolve(action)

    # Verify
    assert resolved.text == "Key is 12345"


@pytest.mark.asyncio
async def test_resolve_target_reference():
    # Setup
    mock_ledger = AsyncMock()
    mock_ledger.get.return_value = "Submit Button"

    service = ReferenceResolutionService(ledger=mock_ledger)

    action = Action(
        action_type=ActionType.TAP,
        rationale="Tap dynamic target",
        target="$memory.button_name",
        natural_language_target="$memory.button_name",
    )

    # Execute
    resolved = await service.resolve(action)

    # Verify
    assert resolved.target == "Submit Button"
    assert resolved.natural_language_target == "Submit Button"


@pytest.mark.asyncio
async def test_resolve_no_reference():
    # Setup
    mock_ledger = AsyncMock()
    service = ReferenceResolutionService(ledger=mock_ledger)

    action = Action(action_type=ActionType.TAP, rationale="Just a tap", target="Submit")

    # Execute
    resolved = await service.resolve(action)

    # Verify
    assert resolved == action
    mock_ledger.get.assert_not_called()
