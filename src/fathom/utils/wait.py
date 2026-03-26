from __future__ import annotations

import asyncio
from logging import getLogger
from typing import TYPE_CHECKING

from fathom.constants.execution import CAPTURE_OVERHEAD_MS, MAX_STABILITY_WAIT_MS

if TYPE_CHECKING:
    from fathom.schemas.configuration import FathomConfiguration

logger = getLogger(__name__)


async def stability_wait(configuration: FathomConfiguration) -> None:
    """Apply a capped post-action stability wait based on engine configuration.

    Subtracts the estimated capture overhead (screenshot I/O) from the requested
    wait so that the total settle time (sleep + capture) stays close to the
    configured value without adding unnecessary latency.
    """
    requested_wait_s = float(configuration.engine.stability_wait)
    requested_wait_ms = requested_wait_s * 1000.0
    capped_wait_ms = min(requested_wait_ms, MAX_STABILITY_WAIT_MS)
    applied_wait_ms = max(0.0, capped_wait_ms - CAPTURE_OVERHEAD_MS)
    stability_wait_s = applied_wait_ms / 1000.0
    logger.info(
        "[WAIT] source=stability_wait requested=%.3fs applied=%.3fs (capture_overhead=%.0fms)",
        requested_wait_s,
        stability_wait_s,
        CAPTURE_OVERHEAD_MS,
    )
    await asyncio.sleep(delay=stability_wait_s)
