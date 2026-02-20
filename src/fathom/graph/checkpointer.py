from __future__ import annotations

import importlib.util
from logging import getLogger
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import MemorySaver

logger = getLogger(__name__)

if TYPE_CHECKING:
    from pathlib import Path


def build_checkpointer(checkpoint_path: "Path") -> Any:
    """Create a persistent checkpointer when available; fallback to in-memory."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if importlib.util.find_spec("langgraph.checkpoint.sqlite") is None:
        logger.warning("LangGraph SQLite checkpointer not installed; using MemorySaver.")
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver as _SqliteSaver

        return _SqliteSaver.from_conn_string(str(checkpoint_path))
    except ImportError:
        try:
            from langgraph.checkpoint.sqlite import SQLiteSaver as _SqliteSaver

            return _SqliteSaver.from_conn_string(str(checkpoint_path))
        except Exception as exc:
            logger.warning("SQLite checkpointer unavailable (%s); falling back to MemorySaver", exc)
    except Exception as exc:
        logger.warning("SqliteSaver failed (%s); falling back to MemorySaver", exc)

    return MemorySaver()
