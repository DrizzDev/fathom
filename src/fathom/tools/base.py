"""Base tool protocols for Fathom.

All tools are defined as Protocols for maximum flexibility and testability.
Implementations can be local, remote, or mock.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Tool(ABC, Generic[T]):
    """Abstract base for all tools.

    Tools are the execution layer of Fathom. They know HOW to perform
    actions, while the agent layer knows WHAT to do.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier.

        Returns:
            Human-readable tool name.
        """
        raise NotImplementedError

    @abstractmethod
    async def execute(self, request: Any) -> T:
        """Execute tool with given request.

        Args:
            request: Tool-specific request payload.

        Returns:
            Tool-specific result.

        Raises:
            ToolError: If execution fails.
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check if tool is available and healthy.

        Returns:
            True if tool is ready for use.
        """
        return True

    async def initialize(self) -> None:
        """Initialize tool resources.

        Called once before first use. Override for lazy initialization.
        """
        pass

    async def cleanup(self) -> None:
        """Clean up tool resources.

        Called when tool is no longer needed. Override for cleanup.
        """
        pass


class ToolProvider(ABC):
    """Abstract base for providing tools to the orchestration layer.

    Tool providers are responsible for creating and managing tool instances.
    They enable dependency injection and support different backends.
    """

    @abstractmethod
    def get_vision_tool(self) -> "VisionTool":
        """Get vision tool for screen analysis.

        Returns:
            Configured vision tool instance.

        Raises:
            ConfigurationError: If tool cannot be created.
        """
        raise NotImplementedError

    @abstractmethod
    def get_device_tool(self) -> "DeviceTool":
        """Get device tool for actions.

        Returns:
            Configured device tool instance.

        Raises:
            ConfigurationError: If tool cannot be created.
        """
        raise NotImplementedError

    @abstractmethod
    def get_capture_tool(self) -> "CaptureTool":
        """Get capture tool for screenshots.

        Returns:
            Configured capture tool instance.

        Raises:
            ConfigurationError: If tool cannot be created.
        """
        raise NotImplementedError


# Import concrete protocols (defined in submodules)
from fathom.tools.capture import CaptureTool  # noqa: E402
from fathom.tools.device import DeviceTool  # noqa: E402
from fathom.tools.vision import VisionTool  # noqa: E402

__all__ = [
    "CaptureTool",
    "DeviceTool",
    "Tool",
    "ToolProvider",
    "VisionTool",
]
