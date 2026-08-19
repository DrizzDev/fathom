from __future__ import annotations

import json
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Optional

import redis.asyncio as redis

from fathom.constants.telemetry import (
    GUARDED_ENVELOPE_KEYS,
    TELEMETRY_COLLISION_COUNTER_START,
    TELEMETRY_COLLISION_PREFIX,
    TelemetryEnvelopeKey,
)
from fathom.interfaces.telemetry import TelemetryPort
from fathom.schemas.configuration import TelemetryConfiguration


class RedisTelemetryAdapter(TelemetryPort):
    """
    Telemetry adapter that publishes logs to Redis for real-time streaming.
    """

    def __init__(
        self,
        *,
        name: str = "fathom",
        configuration: TelemetryConfiguration,
        client: Optional[redis.Redis] = None,
    ) -> None:
        """
        Initialize Redis telemetry adapter from configuration with an optional injected client.
        """

        if not configuration.connection_string:
            raise ValueError("Redis telemetry requires 'connection_string'.")

        if not configuration.topic:
            raise ValueError("Redis telemetry requires 'topic'.")

        if not configuration.session_id:
            raise ValueError("Redis telemetry requires 'session_id'.")

        self.__configuration = configuration
        self.__identity = configuration.identity
        self.__session_id = configuration.session_id
        self.__channel = configuration.topic.format(
            session_id=self.__session_id, identity=self.__identity
        )
        self.__redis = (
            client
            if client is not None
            else redis.from_url(configuration.connection_string, decode_responses=True)
        )

        self.__logger = getLogger(name=name)

    def update_identity(self, identity: str) -> None:
        """
        Update the identity used for routing.
        """

        self.__identity = identity

        if self.__configuration.topic:
            self.__channel = self.__configuration.topic.format(
                session_id=self.__session_id, identity=self.__identity
            )

    async def __publish(
        self,
        *,
        message: str,
        level: str,
        color: str,
        context: Dict[str, Any],
    ) -> None:
        """
        Publish log event to Redis matching the farm-wrap gateway schema.
        """

        try:
            safe_context = self.__rename_reserved(context=context)
            payload = {
                TelemetryEnvelopeKey.EVENT.value: "log",
                TelemetryEnvelopeKey.LEVEL.value: level,
                TelemetryEnvelopeKey.COLOR.value: color,
                TelemetryEnvelopeKey.SOURCE.value: "fathom",
                TelemetryEnvelopeKey.MESSAGE.value: message,
                TelemetryEnvelopeKey.REQUEST_ID.value: self.__identity,
                TelemetryEnvelopeKey.SESSION_ID.value: self.__session_id,
                TelemetryEnvelopeKey.TIMESTAMP.value: datetime.now()
                .astimezone()
                .strftime("%Y-%m-%d %H:%M:%S %Z"),
                **safe_context,
            }
            await self.__redis.publish(self.__channel, json.dumps(payload))
        except Exception:
            self.__logger.warning("Failed to publish telemetry to Redis", exc_info=True)

    def __rename_reserved(self, *, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rename caller-supplied keys that would collide with envelope keys.
        """

        if not (collisions := {key.value for key in GUARDED_ENVELOPE_KEYS} & context.keys()):
            return context

        self.__logger.warning(
            "Telemetry context collided with reserved envelope keys",
            extra={"collisions": sorted(collisions)},
        )

        safe = dict(context)
        for original in collisions:
            renamed = self.__unique_key(
                existing=safe,
                base=f"{TELEMETRY_COLLISION_PREFIX}{original}",
            )
            safe[renamed] = safe.pop(original)

        return safe

    @staticmethod
    def __unique_key(*, base: str, existing: Dict[str, Any]) -> str:
        """
        Return a non-colliding key by appending an incrementing counter when needed.
        """

        if base not in existing:
            return base

        counter = TELEMETRY_COLLISION_COUNTER_START

        while f"{base}_{counter}" in existing:
            counter += 1

        return f"{base}_{counter}"

    @staticmethod
    def __error_context(
        *, context: Dict[str, Any], exception: Optional[BaseException]
    ) -> Dict[str, Any]:
        """
        Build structured telemetry context for an error event.
        """

        if exception is None:
            return context

        return {
            **context,
            "exception_type": type(exception).__name__,
        }

    async def debug(self, message: str, **context: Any) -> None:
        """Log at debug level and publish to the telemetry channel."""

        self.__logger.debug(message, extra=context)
        await self.__publish(message=message, level="debug", color="gray", context=context)

    async def info(self, message: str, **context: Any) -> None:
        """Log at info level and publish to the telemetry channel."""

        self.__logger.info(message, extra=context)
        await self.__publish(message=message, level="info", color="blue", context=context)

    async def warning(self, message: str, **context: Any) -> None:
        """Log at warning level and publish to the telemetry channel."""

        self.__logger.warning(message, extra=context)
        await self.__publish(message=message, level="warning", color="yellow", context=context)

    async def error(self, message: str, **context: Any) -> None:
        """Log at error level and publish to the telemetry channel."""

        self.__logger.error(message, extra=context)
        await self.__publish(message=message, level="error", color="red", context=context)

    async def exception(
        self,
        message: str,
        *,
        exception: Optional[BaseException] = None,
        **context: Any,
    ) -> None:
        """
        Publishes ERROR logs together with exception details.
        """

        payload = self.__error_context(exception=exception, context=context)

        if exception is None:
            self.__logger.exception(message, extra=payload)
        else:
            self.__logger.error(
                message,
                extra=payload,
                exc_info=(type(exception), exception, exception.__traceback__),
            )

        await self.__publish(message=message, level="error", color="red", context=payload)

    async def close(self) -> None:
        """
        Close Redis connection.
        """

        await self.__redis.aclose()
