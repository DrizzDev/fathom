from __future__ import annotations

import asyncio
from logging import getLogger
from typing import TYPE_CHECKING

from fathom.constants.execution import MAX_STABILITY_WAIT_MS

if TYPE_CHECKING:
    from fathom.schemas.configuration import FathomConfiguration

logger = getLogger(__name__)


async def stability_wait(configuration: FathomConfiguration) -> None:
    """Apply a capped post-action stability wait based on engine configuration."""
    requested_wait_s = float(configuration.engine.stability_wait)
    requested_wait_ms = requested_wait_s * 1000.0
    applied_wait_ms = min(requested_wait_ms, MAX_STABILITY_WAIT_MS)
    stability_wait_s = applied_wait_ms / 1000.0
    logger.debug(
        "[WAIT] source=stability_wait requested=%.3fs applied=%.3fs",
        requested_wait_s,
        stability_wait_s,
    )
    await asyncio.sleep(delay=stability_wait_s)
