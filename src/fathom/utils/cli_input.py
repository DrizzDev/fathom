from __future__ import annotations

import select
import sys
import threading
from typing import Optional

_INPUT_LOCK = threading.Lock()


def input_with_lock(prompt: str) -> str:
    """Read a line from stdin without colliding with other readers."""

    _INPUT_LOCK.acquire()
    try:
        return input(prompt)
    finally:
        _INPUT_LOCK.release()


def poll_input_line() -> Optional[str]:
    """Non-blocking stdin line read. Returns None if no input is ready."""

    if not _INPUT_LOCK.acquire(blocking=False):
        return None
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.readline()
    finally:
        _INPUT_LOCK.release()
