"""
Exploration-based execution strategy using hexagonal architecture.

Migrated from agent/strategies/exploration.py with ports instead of tools.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from logging import getLogger
from typing import Any, Dict, Optional

from fathom.constants.execution import VISUAL_HASH_LENGTH
from fathom.core.context.manager import ContextManager
from fathom.core.exceptions import StrategyError
from fathom.core.execution.engine import ExecutionEngine
from fathom.interfaces.device import DevicePort
from fathom.interfaces.storage import StoragePort
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.actions import Action
from fathom.schemas.exploration import ActionGenerator, ExplorationGraph
from fathom.schemas.results import ExecutionResult
from fathom.schemas.screens import ScreenCapture, ScreenState
from fathom.schemas.steps import Step

logger = getLogger(name=__name__)


class ExplorationStrategy:
    """
    Strategy for autonomous application mapping using hexagonal architecture.

    This is the REAL implementation migrated from agent/strategies/exploration.py
    but adapted to use ports (DevicePort, StoragePort, etc.) instead of tools.
    """

    def __init__(
        self,
        engine: ExecutionEngine,
        context: ContextManager,
        *,
        device: DevicePort,
        storage: StoragePort,
        telemetry: TelemetryPort,
        max_steps: int = 100,
        timeout: float = 3600.0,
        seed: Optional[int] = None,
        package_name: str = "unknown_app",
        workflow_id: str = "exploration",
    ) -> None:
        """
        Initialize exploration strategy with ports.

        Args:
            engine: Execution engine for running steps
            context: Context manager for tracking execution state
            device: Device port for screen capture and device info
            storage: Storage port for saving screenshots
            telemetry: Telemetry port for logging
            max_steps: Maximum number of exploration steps
            timeout: Maximum exploration time in seconds
            seed: Random seed for reproducible exploration
            package_name: Target package name
            workflow_id: Execution session ID
        """
        self.__engine = engine
        self.__context = context
        self.__package_name = package_name
        self.__workflow_id = workflow_id

        # Ports
        self.__device = device
        self.__storage = storage
        self.__telemetry = telemetry

        # Configuration
        self.__max_steps = max_steps
        self.__timeout = timeout

        # Exploration state
        self.__steps = 0
        self.__start = time.time()
        self.__graph = ExplorationGraph()
        self.__generator = ActionGenerator(seed=seed)

        # Tracking
        self.__last: Optional[Action] = None
        self.__current: Optional[ScreenState] = None

        # Cancellation support
        self.__cancelled = False

    @property
    def graph(self) -> ExplorationGraph:
        """Exploration graph."""
        return self.__graph

    async def execute(self, max_steps: int) -> ExecutionResult:
        """
        Execute exploration workflow.

        This is the REAL execution loop from agent/strategies/exploration.py
        """
        start_time = time.time()
        self.__context.set_roadmap(intent="Explore application structure")

        success = False
        error = None

        try:
            # Main exploration loop
            while await self.__should_continue() and not self.__cancelled:
                if self.__steps >= max_steps:
                    break

                # Execute one exploration step
                step_success = await self.__execute_step()

                if not step_success:
                    # Continue even if individual step fails
                    self.__telemetry.warning("Exploration step failed, continuing")

            if self.__cancelled:
                error = "Exploration cancelled by user"
                success = False
            else:
                success = True

        except StrategyError as exception:
            logger.exception(f"Exploration strategy execution failed: {exception}")
            error = str(exception)
            success = False
        except Exception as exception:
            logger.exception(f"Unexpected error in exploration strategy: {exception}")
            raise StrategyError("Exploration execution failed unexpectedly") from exception

        duration = int((time.time() - start_time) * 1000)

        return ExecutionResult(
            success=success,
            duration=duration,
            error=error,
        )

    async def __execute_step(self) -> bool:
        """
        Execute one discovery step.

        Migrated from agent/strategies/exploration.py
        """
        try:
            # 1. Capture screen
            capture = await self.__capture_screen()
            if not capture:
                return False

            # 2. Compute screen state
            state = await self.__compute_state(capture=capture)
            fingerprint = state.visual_hash

            # 3. Update graph
            node = self.__graph.add_screen(state=state)

            # 4. Record transition from previous screen
            if self.__last and self.__current:
                self.__graph.record_transition(
                    destination=fingerprint,
                    origin=self.__current.visual_hash,
                    action=self.__last.to_description(),
                )

            self.__current = state

            # 5. Generate exploratory action
            size = await self.__device.get_screen_size()
            action = self.__generator.generate(node=node, width=size[0], height=size[1])

            # 6. Create step
            step = Step(
                action=action,
                screen_hash=fingerprint,
                step_number=self.__steps,
            )

            # 7. Execute through engine
            result = await self.__engine.execute_step(
                step=step,
                package_name=self.__package_name,
                session_id=self.__workflow_id,
            )

            self.__steps += 1
            self.__last = action

            # 8. Stability wait
            await asyncio.sleep(delay=0.5)

            # 9. Capture post-action screen
            post_capture = await self.__capture_screen()
            if post_capture:
                post_state = await self.__compute_state(capture=post_capture)
                screen_changed = fingerprint != post_state.visual_hash

                # Update context
                await self.__context.commit(
                    observation=f"Screen: {state.activity} -> {post_state.activity if screen_changed else 'same'}",
                    thought=action.rationale,
                    action=action,
                )

            return result.success

        except StrategyError as exception:
            logger.exception(f"Exploration step failed: {exception}")
            return False
        except Exception as exception:
            logger.exception(f"Unexpected error in exploration step: {exception}")
            raise StrategyError("Exploration step failed unexpectedly") from exception

    async def __capture_screen(self) -> Optional[ScreenCapture]:
        """Capture screen through device port."""
        try:
            screenshot_bytes = await self.__device.capture_screen()

            # Get screen dimensions
            width, height = await self.__device.get_screen_size()

            # Get current activity
            try:
                activity = await self.__device.get_current_package()
            except Exception:
                activity = "unknown"

            # Store screenshot
            storage_id = await self.__storage.save(
                data=screenshot_bytes,
                metadata={"type": "screenshot", "timestamp": time.time()},
            )

            return ScreenCapture(
                width=width,
                height=height,
                activity=activity,
                image=screenshot_bytes,
                timestamp=int(time.time() * 1000),
                metadata={"storage_id": storage_id},
            )

        except StrategyError as exception:
            logger.exception(f"Screen capture failed: {exception}")
            return None
        except Exception as exception:
            logger.exception(f"Unexpected error in screen capture: {exception}")
            raise StrategyError("Screen capture failed unexpectedly") from exception

    async def __compute_state(self, capture: ScreenCapture) -> ScreenState:
        """Compute screen state from capture."""
        visual_hash = hashlib.sha256(capture.image).hexdigest()[:VISUAL_HASH_LENGTH]

        return ScreenState(
            visual_hash=visual_hash,
            activity=capture.activity,
            timestamp=capture.timestamp,
            activity_hash=hashlib.md5(capture.activity.encode(), usedforsecurity=False).hexdigest()[
                :VISUAL_HASH_LENGTH
            ],
            structural_hash="0" * VISUAL_HASH_LENGTH,
        )

    async def __should_continue(self) -> bool:
        """
        Stop conditions.

        Migrated from agent/strategies/exploration.py
        """
        if self.__steps >= self.__max_steps:
            return False

        return (time.time() - self.__start) < self.__timeout

    def get_progress(self) -> Dict[str, Any]:
        """
        Discovery metrics.

        Migrated from agent/strategies/exploration.py
        """
        return {
            "steps": self.__steps,
            "stats": self.__graph.get_stats(),
            "elapsed": time.time() - self.__start,
            "context": self.__context.get_full_context(),
        }

    def cancel(self) -> None:
        """Cancel the exploration."""
        self.__cancelled = True
        self.__telemetry.warning("Exploration strategy cancellation requested")
