from __future__ import annotations

import io
import time
from typing import Any, Dict

import pytest
from PIL import Image, ImageDraw

from fathom.constants import ActionType
from fathom.core.context.manager import ContextManager
from fathom.core.services.vision import VisionService
from fathom.interfaces.llm import LLMPort
from fathom.interfaces.memory import MemoryPort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.screens import ScreenCapture

pytestmark = pytest.mark.release

EMAIL = "dev+test+Ilu+z2O5@varomoney.com"


class NoopTelemetry(TelemetryPort):
    """
    Telemetry sink for live LLM service tests.
    """

    async def debug(self, message: str, **context: Any) -> None:
        """
        Drop debug telemetry.
        """

        _ = message, context

    async def info(self, message: str, **context: Any) -> None:
        """
        Drop info telemetry.
        """

        _ = message, context

    async def warning(self, message: str, **context: Any) -> None:
        """
        Drop warning telemetry.
        """

        _ = message, context

    async def error(self, message: str, **context: Any) -> None:
        """
        Drop error telemetry.
        """

        _ = message, context

    async def exception(
        self,
        message: str,
        *,
        exception: BaseException | None = None,
        **context: Any,
    ) -> None:
        """
        Drop exception telemetry.
        """

        _ = message, exception, context


class LoginScreenFixture:
    """
    Builds a small visual login screen fixture for live planner calls.
    """

    @staticmethod
    def capture() -> ScreenCapture:
        """
        Return a synthetic login screen capture.
        """

        image = Image.new("RGB", (402, 874), "white")
        draw = ImageDraw.Draw(image)
        draw.text((38, 80), "Varo", fill="black")
        draw.text((38, 140), "Email", fill="black")
        draw.rectangle((38, 170, 364, 225), outline="black", width=2)
        draw.text((50, 188), EMAIL, fill="black")
        draw.rectangle((38, 270, 364, 330), fill="#1167ff")
        draw.text((160, 292), "Continue", fill="white")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        return ScreenCapture(
            width=402,
            height=874,
            activity="com.varomoney.varo",
            image=buffer.getvalue(),
            timestamp=int(time.time() * 1000),
        )

    @staticmethod
    def elements() -> Dict[str, Dict[str, str]]:
        """
        Return a manifest matching the synthetic login screen.
        """

        return {
            "1": {
                "type": "XCUIElementTypeTextField",
                "label": "Email",
                "value": EMAIL,
                "bounds": "[38,170][364,225]",
            },
            "2": {
                "type": "XCUIElementTypeButton",
                "label": "Continue",
                "bounds": "[38,270][364,330]",
            },
        }


class TestVisionService:
    """
    Live LLM checks for VisionService planner behavior.
    """

    async def test_user_guidance_prevents_retyping_filled_email(
        self,
        llm: LLMPort,
        memory_port_stub: MemoryPort,
    ) -> None:
        """
        Human guidance must make the planner tap Continue instead of retyping email.
        """

        context_manager = ContextManager(memory=memory_port_stub, workflow_id="live-vision")
        await context_manager.inject_user_guidance(
            guidance=f"The email field already contains {EMAIL}. Do not type it again; tap Continue.",
            step=1,
        )
        service = VisionService(
            llm=llm,
            memory=memory_port_stub,
            telemetry=NoopTelemetry(),
            use_cache=False,
            session_id="live-vision",
        )

        analysis = await service.analyze(
            intent="Continue login after the email address is filled.",
            capture=LoginScreenFixture.capture(),
            context_manager=context_manager,
            visual_hash="abcdef1234567890",
            screen_width=402,
            screen_height=874,
            use_xml=True,
            elements=LoginScreenFixture.elements(),
        )

        assert analysis.action.action_type == ActionType.TAP
        assert analysis.action.text != EMAIL
        assert (
            analysis.action.label_id == "2"
            or "continue" in (analysis.action.target or "").casefold()
        )
