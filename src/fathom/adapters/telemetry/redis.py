from __future__ import annotations

import json
from datetime import datetime
from logging import getLogger
from typing import Any

import redis.asyncio as redis

from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import TelemetryConfiguration


class RedisTelemetryAdapter(TelemetryPort):
    """
    Telemetry adapter that publishes logs to Redis for real-time streaming.
    """

    def __init__(self, configuration: TelemetryConfiguration, logger_name: str = "fathom") -> None:
        """
        Initialize Redis telemetry adapter using configuration object.
        """

        if not configuration.connection_string:
            raise ValueError("Redis telemetry requires 'connection_string'.")

        if not configuration.topic:
            raise ValueError("Redis telemetry requires 'topic'.")

        if not configuration.session_id:
            raise ValueError("Redis telemetry requires 'session_id'.")

        self.__session_id = configuration.session_id
        self.__channel = configuration.topic.format(session_id=self.__session_id)
        self.__redis = redis.from_url(configuration.connection_string, decode_responses=True)

        self.__logger = getLogger(name=logger_name)

    async def __publish(self, level: str, message: str, color: str, **context: Any) -> None:
        """
        Publish log event to Redis matching the farm-wrap gateway schema.
        """

        try:
            payload = {
                "event": "log",
                "level": level,
                "color": color,
                "source": "fathom",
                "message": message,
                "session_id": self.__session_id,
                "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                **context,
            }
            await self.__redis.publish(self.__channel, json.dumps(payload))
        except Exception:
            self.__logger.warning("Failed to publish telemetry to Redis", exc_info=True)

    async def debug(self, message: str, **context: Any) -> None:
        """
        Publishes DEBUG Logs
        """

        self.__logger.debug(message, extra=context)
        await self.__publish("debug", message, "gray", **context)

    async def info(self, message: str, **context: Any) -> None:
        """
        Publishes INFO Logs
        """

        self.__logger.info(message, extra=context)
        await self.__publish("info", message, "blue", **context)

    async def warning(self, message: str, **context: Any) -> None:
        """
        Publishes WARNING Logs
        """

        self.__logger.warning(message, extra=context)
        await self.__publish("warning", message, "yellow", **context)

    async def error(self, message: str, **context: Any) -> None:
        """
        Publishes ERROR Logs
        """

        self.__logger.error(message, extra=context)
        await self.__publish("error", message, "red", **context)

    async def close(self) -> None:
        """
        Close Redis connection.
        """

        await self.__redis.aclose()
