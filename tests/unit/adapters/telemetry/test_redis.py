from __future__ import annotations

import json
import unittest
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock

from fathom.adapters.telemetry.redis import RedisTelemetryAdapter
from fathom.schemas.configuration import TelemetryConfiguration


class TestRedisTelemetryReservedKeyGuard(unittest.IsolatedAsyncioTestCase):
    """
    Caller-supplied context keys must not silently overwrite envelope keys.
    """

    def setUp(self) -> None:
        """
        Build a RedisTelemetryAdapter with a mocked redis client that captures publishes.
        """

        configuration = TelemetryConfiguration(
            type="REDIS",
            connection_string="redis://localhost:6379/0",
            topic="fathom:logs:{session_id}",
            session_id="session-1",
            identity="workflow-1",
        )
        self.__published: List[Tuple[str, str]] = []
        publisher = AsyncMock(side_effect=self.__capture_publish)
        self.__client = MagicMock(publish=publisher, aclose=AsyncMock())
        self.__adapter: RedisTelemetryAdapter = RedisTelemetryAdapter(
            configuration=configuration,
            client=self.__client,
        )

    async def __capture_publish(self, channel: str, payload: str) -> None:
        """
        Record one publish call for later assertions in the active test.
        """

        self.__published.append((channel, payload))

    async def asyncTearDown(self) -> None:
        """
        Close the redis client after each test.
        """

        await self.__adapter.close()

    def __last_payload(self) -> dict:
        """
        Return the JSON-decoded payload from the most recent publish.
        """

        _, payload = self.__published[-1]
        return json.loads(payload)

    async def test_event_collision_is_renamed_to_context_event(self) -> None:
        """
        A caller-supplied `event` must be renamed; the envelope `event` stays `log`.
        """

        await self.__adapter.info("hi", event="planner.ask_user.emitted")

        payload = self.__last_payload()
        self.assertEqual(payload["event"], "log")
        self.assertEqual(payload["context_event"], "planner.ask_user.emitted")

    async def test_secondary_collision_appends_counter_suffix(self) -> None:
        """
        If both `event` and `context_event` are supplied, the second collision must be suffixed.
        """

        await self.__adapter.info("hi", event="first", context_event="second")

        payload = self.__last_payload()
        self.assertEqual(payload["event"], "log")
        self.assertEqual(payload["context_event"], "second")
        self.assertEqual(payload["context_event_2"], "first")

    async def test_source_collision_is_renamed(self) -> None:
        """
        A caller-supplied `source` must not overwrite the envelope source.
        """

        await self.__adapter.warning("hi", source="caller")

        payload = self.__last_payload()
        self.assertEqual(payload["source"], "fathom")
        self.assertEqual(payload["context_source"], "caller")

    async def test_non_colliding_context_passes_through_unchanged(self) -> None:
        """
        Context keys outside the reserved set must appear as-is on the envelope.
        """

        await self.__adapter.info("hi", operation="record_run_started", count=3)

        payload = self.__last_payload()
        self.assertEqual(payload["operation"], "record_run_started")
        self.assertEqual(payload["count"], 3)

    async def test_debug_publish_also_uses_guard(self) -> None:
        """
        Debug-level publish must also rename reserved keys (not only info/warning/error).
        """

        await self.__adapter.debug("hi", requestId="caller-supplied")

        payload = self.__last_payload()
        self.assertEqual(payload["requestId"], "workflow-1")
        self.assertEqual(payload["context_requestId"], "caller-supplied")

    async def test_message_collision_is_renamed(self) -> None:
        """
        Caller-supplied `message` is renamed; the envelope `message` reflects the log text.
        """

        publish = self.__adapter._RedisTelemetryAdapter__publish
        await publish(
            message="hi",
            level="info",
            color="green",
            context={"message": "caller-supplied"},
        )

        payload = self.__last_payload()
        self.assertEqual(payload["message"], "hi")
        self.assertEqual(payload["context_message"], "caller-supplied")

    async def test_level_collision_is_renamed(self) -> None:
        """
        Caller-supplied `level` is renamed; the envelope `level` reflects the helper.
        """

        await self.__adapter.warning("hi", level="caller-supplied")

        payload = self.__last_payload()
        self.assertEqual(payload["level"], "warning")
        self.assertEqual(payload["context_level"], "caller-supplied")

    async def test_collision_is_not_logged_as_error(self) -> None:
        """
        Recovered reserved-key collisions must not create error-level telemetry noise.
        """

        with self.assertLogs("fathom", level="WARNING") as captured:
            await self.__adapter.info("hi", session_id="caller-supplied")

        payload = self.__last_payload()
        self.assertEqual(payload["session_id"], "session-1")
        self.assertEqual(payload["context_session_id"], "caller-supplied")
        self.assertTrue(
            any(
                "Telemetry context collided with reserved envelope keys" in message
                for message in captured.output
            )
        )
        self.assertFalse(any("ERROR" in message for message in captured.output))

    async def test_color_collision_is_renamed(self) -> None:
        """
        Caller-supplied `color` is renamed; the envelope `color` reflects the helper.
        """

        await self.__adapter.error("hi", color="caller-supplied")

        payload = self.__last_payload()
        self.assertEqual(payload["color"], "red")
        self.assertEqual(payload["context_color"], "caller-supplied")
