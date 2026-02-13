from __future__ import annotations

import pytest

from fathom.constants import ActionType
from fathom.orchestration.executor import StepExecutor
from fathom.schemas.actions import Action, Bounds
from fathom.schemas.orchestration import ExecutionContext
from fathom.schemas.steps import Step
from fathom.tools.capture.mock import MockCaptureTool
from fathom.tools.device.mock import MockDeviceTool


@pytest.mark.asyncio
async def test_type_requires_bounds_for_focus_guard() -> None:
    device = MockDeviceTool()
    capture = MockCaptureTool()
    executor = StepExecutor(device=device, capture=capture)
    context = ExecutionContext(workflow_id="typing-guard")

    step = Step(
        step_number=1,
        screen_hash="pre",
        action=Action(
            action_type=ActionType.TYPE,
            target="search field",
            text="hello",
            rationale="Type query text",
        ),
    )

    result = await executor.execute(step=step, context=context)

    assert not result.success
    assert result.error == "Type action requires bounds for focus tap guard"
    assert device.tap_calls == []
    assert device.type_calls == []


@pytest.mark.asyncio
async def test_type_taps_before_typing_when_bounds_present() -> None:
    device = MockDeviceTool(screen_size=(1000, 1000))
    capture = MockCaptureTool(screen_width=1000, screen_height=1000)
    executor = StepExecutor(device=device, capture=capture)
    context = ExecutionContext(workflow_id="typing-guard")

    step = Step(
        step_number=1,
        screen_hash="pre",
        action=Action(
            action_type=ActionType.TYPE,
            target="search field",
            text="hello",
            rationale="Type query text",
            bounds=Bounds(x=100, y=200, width=300, height=80),
        ),
    )

    result = await executor.execute(step=step, context=context)

    assert result.success
    assert result.error is None
    assert device.tap_calls == [(250, 240)]
    assert device.type_calls == ["hello"]
