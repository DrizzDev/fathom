from __future__ import annotations

import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

from fathom.adapters.ios.idb import IDBClient
from fathom.core.exceptions import DeviceError


class _FakeProcess:
    """
    Minimal stand-in for ``asyncio.subprocess.Process`` that returns
    canned stdout/stderr/return_code via ``communicate``.
    """

    def __init__(
        self,
        *,
        return_code: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.returncode: int = return_code
        self.__stdout = stdout
        self.__stderr = stderr

    async def communicate(self) -> Tuple[bytes, bytes]:
        return self.__stdout, self.__stderr

    def kill(self) -> None:  # pragma: no cover - safety only
        return None

    async def wait(self) -> int:  # pragma: no cover - safety only
        return self.returncode


def _patch_subprocess(
    *,
    processes: List[_FakeProcess],
) -> Tuple[Any, List[List[str]], List[Optional[Dict[str, str]]]]:
    """
    Patch ``asyncio.create_subprocess_exec`` to yield queued processes
    and capture the argv + env each invocation received.
    """

    captured_argv: List[List[str]] = []
    captured_env: List[Optional[Dict[str, str]]] = []
    iterator = iter(processes)

    async def fake_create(*args: object, **kwargs: object) -> _FakeProcess:
        captured_argv.append([str(arg) for arg in args])
        env = kwargs.get("env")
        captured_env.append(env if isinstance(env, dict) or env is None else None)
        return next(iterator)

    return (
        patch("fathom.adapters.ios.idb.asyncio.create_subprocess_exec", side_effect=fake_create),
        captured_argv,
        captured_env,
    )


class IDBClientArgvTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify each public method assembles the right ``idb`` argv tail
    and forwards ``--udid`` when the client is bound to a target.
    """

    async def __invoke(
        self,
        *,
        client_kwargs: Dict[str, Any],
        method: str,
        method_kwargs: Dict[str, Any],
        process: _FakeProcess,
    ) -> Tuple[List[str], Optional[Dict[str, str]]]:
        client = IDBClient(**client_kwargs)
        patcher, captured_argv, captured_env = _patch_subprocess(processes=[process])
        with patcher:
            await getattr(client, method)(**method_kwargs)
        return captured_argv[0], captured_env[0]

    async def test_tap_argv_does_not_carry_udid(self) -> None:
        """``--udid`` lives in IDB_UDID env, not argv."""

        argv, env = await self.__invoke(
            client_kwargs={"udid": "DEVICE-1"},
            method="tap",
            method_kwargs={"x": 100.6, "y": 200.4},
            process=_FakeProcess(return_code=0),
        )
        self.assertEqual(argv, ["idb", "ui", "tap", "101", "200"])
        self.assertNotIn("--udid", argv)
        assert env is not None
        self.assertEqual(env.get("IDB_UDID"), "DEVICE-1")

    async def test_swipe_argv_emits_duration_in_seconds(self) -> None:
        argv, _env = await self.__invoke(
            client_kwargs={"udid": "DEVICE-1"},
            method="swipe",
            method_kwargs={"x1": 0, "y1": 0, "x2": 100, "y2": 200, "duration_milliseconds": 750},
            process=_FakeProcess(return_code=0),
        )
        self.assertIn("--duration", argv)
        duration_index = argv.index("--duration")
        self.assertEqual(argv[duration_index + 1], "0.750")

    async def test_press_home_argv(self) -> None:
        argv, _env = await self.__invoke(
            client_kwargs={"udid": "DEVICE-1"},
            method="press_home",
            method_kwargs={},
            process=_FakeProcess(return_code=0),
        )
        self.assertEqual(argv[-3:], ["ui", "button", "HOME"])

    async def test_launch_argv(self) -> None:
        argv, _env = await self.__invoke(
            client_kwargs={"udid": "DEVICE-1"},
            method="launch",
            method_kwargs={"bundle_identifier": "com.example.app"},
            process=_FakeProcess(return_code=0),
        )
        self.assertEqual(argv[-2:], ["launch", "com.example.app"])

    async def test_omits_udid_env_when_unset(self) -> None:
        argv, env = await self.__invoke(
            client_kwargs={"udid": None},
            method="tap",
            method_kwargs={"x": 10, "y": 10},
            process=_FakeProcess(return_code=0),
        )
        self.assertNotIn("--udid", argv)
        # No env override when the client has no UDID — subprocess
        # inherits the parent's environment unmodified.
        self.assertIsNone(env)


class IDBClientResultTest(unittest.IsolatedAsyncioTestCase):
    """
    Verify return parsing + error mapping for the read-only methods.
    """

    async def test_capture_screen_writes_to_tempfile_then_reads_bytes(self) -> None:
        """``idb screenshot <path>`` writes the PNG to a file; the
        client reads it back and returns the bytes."""

        from pathlib import Path

        client = IDBClient(udid="DEVICE-1")
        png = b"\x89PNG\r\n\x1a\nfake-payload"

        captured_paths: List[str] = []

        async def fake_create(*args: object, **_kwargs: object) -> _FakeProcess:
            argv = [str(arg) for arg in args]
            # idb writes the PNG to the path passed as the last argv slot.
            tmp_path = argv[-1]
            captured_paths.append(tmp_path)
            Path(tmp_path).write_bytes(png)
            return _FakeProcess(return_code=0)

        with patch(
            "fathom.adapters.ios.idb.asyncio.create_subprocess_exec",
            side_effect=fake_create,
        ):
            result = await client.capture_screen()

        self.assertEqual(result, png)
        # Tempfile must be cleaned up after the read.
        self.assertEqual(len(captured_paths), 1)
        self.assertFalse(Path(captured_paths[0]).exists())

    async def test_capture_screen_raises_when_tempfile_empty(self) -> None:
        """A successful idb invocation that wrote nothing must surface
        as ``DeviceError`` — we never want to feed an empty payload to
        the dimension cache or downstream perception."""

        client = IDBClient(udid="DEVICE-1")

        async def fake_create(*_args: object, **_kwargs: object) -> _FakeProcess:
            return _FakeProcess(return_code=0)

        with (
            patch(
                "fathom.adapters.ios.idb.asyncio.create_subprocess_exec",
                side_effect=fake_create,
            ),
            self.assertRaises(DeviceError),
        ):
            await client.capture_screen()

    async def test_list_apps_parses_ndjson(self) -> None:
        client = IDBClient(udid="DEVICE-1")
        ndjson = (
            b'{"bundle_id": "com.example.beta", "name": "Beta"}\n'
            b'{"bundle_id": "com.example.alpha", "name": "Alpha"}\n'
            b"\n"
            b'{"not_a_bundle": true}\n'
            b"junk-not-json\n"
        )
        patcher, _captured, _env = _patch_subprocess(
            processes=[_FakeProcess(return_code=0, stdout=ndjson)],
        )
        with patcher:
            bundles = await client.list_apps()
        self.assertEqual(bundles, ["com.example.alpha", "com.example.beta"])

    async def test_list_apps_returns_empty_on_failure(self) -> None:
        client = IDBClient(udid="DEVICE-1")
        patcher, _captured, _env = _patch_subprocess(
            processes=[_FakeProcess(return_code=2, stderr=b"no companion")],
        )
        with patcher:
            bundles = await client.list_apps()
        self.assertEqual(bundles, [])

    async def test_dump_source_raises_device_error(self) -> None:
        client = IDBClient(udid="DEVICE-1")
        with self.assertRaises(DeviceError):
            await client.dump_source()

    async def test_failed_command_raises_device_error_with_stderr(self) -> None:
        client = IDBClient(udid="DEVICE-1")
        patcher, _captured, _env = _patch_subprocess(
            processes=[_FakeProcess(return_code=2, stderr=b"target not found")],
        )
        with patcher, self.assertRaises(DeviceError) as ctx:
            await client.tap(x=10, y=10)
        self.assertIn("target not found", str(ctx.exception))

    async def test_missing_idb_executable_raises_helpful_error(self) -> None:
        client = IDBClient(udid="DEVICE-1")

        async def boom(*_args: object, **_kwargs: object) -> _FakeProcess:
            raise FileNotFoundError("idb")

        with (
            patch(
                "fathom.adapters.ios.idb.asyncio.create_subprocess_exec",
                side_effect=boom,
            ),
            self.assertRaises(DeviceError) as ctx,
        ):
            await client.tap(x=10, y=10)
        self.assertIn("idb CLI not found", str(ctx.exception))


class IDBClientDescribeTest(unittest.IsolatedAsyncioTestCase):
    """
    ``describe`` parses point dimensions from ``idb describe --json`` and
    caches the result — ``IOSDevice`` relies on it for pixel→point
    coordinate conversion before every tap/swipe, so a single idb
    shell-out must cover an entire session's gestures.
    """

    def __describe_payload(self) -> bytes:
        return (
            b'{"udid": "DEVICE-1", "screen_dimensions": {'
            b'"width": 1206, "height": 2622, "density": 3.0, '
            b'"width_points": 402, "height_points": 874}}'
        )

    async def test_describe_parses_point_dimensions(self) -> None:
        client = IDBClient(udid="DEVICE-1")
        patcher, captured, _env = _patch_subprocess(
            processes=[_FakeProcess(return_code=0, stdout=self.__describe_payload())],
        )
        with patcher:
            result = await client.describe()

        self.assertEqual(result, (402, 874))
        # Argv must explicitly request JSON so we don't fall back to
        # parsing the pretty-printed Python repr form.
        self.assertEqual(captured[0][-2:], ["describe", "--json"])

    async def test_describe_caches_across_calls(self) -> None:
        """A second ``describe`` call must reuse the cached result —
        screen geometry is stable for the life of the companion
        binding, so we don't want to shell out on every tap."""

        client = IDBClient(udid="DEVICE-1")
        patcher, captured, _env = _patch_subprocess(
            processes=[_FakeProcess(return_code=0, stdout=self.__describe_payload())],
        )
        with patcher:
            first = await client.describe()
            second = await client.describe()

        self.assertEqual(first, second)
        self.assertEqual(len(captured), 1)

    async def test_set_udid_invalidates_cached_dimensions(self) -> None:
        """Retargeting to a new UDID must drop the cached dims so the
        next describe hits the new device's geometry."""

        client = IDBClient(udid="DEVICE-1")
        patcher, captured, _env = _patch_subprocess(
            processes=[
                _FakeProcess(return_code=0, stdout=self.__describe_payload()),
                _FakeProcess(
                    return_code=0,
                    stdout=(
                        b'{"screen_dimensions": {"width": 750, "height": 1334, '
                        b'"density": 2.0, "width_points": 375, "height_points": 667}}'
                    ),
                ),
            ],
        )
        with patcher:
            await client.describe()
            client.set_udid(udid="DEVICE-2")
            second = await client.describe()

        self.assertEqual(second, (375, 667))
        self.assertEqual(len(captured), 2)

    async def test_describe_raises_on_missing_screen_block(self) -> None:
        client = IDBClient(udid="DEVICE-1")
        patcher, _captured, _env = _patch_subprocess(
            processes=[_FakeProcess(return_code=0, stdout=b'{"udid": "DEVICE-1"}')],
        )
        with patcher, self.assertRaises(DeviceError) as ctx:
            await client.describe()
        self.assertIn("screen_dimensions", str(ctx.exception))

    async def test_describe_raises_on_non_integer_points(self) -> None:
        client = IDBClient(udid="DEVICE-1")
        patcher, _captured, _env = _patch_subprocess(
            processes=[
                _FakeProcess(
                    return_code=0,
                    stdout=(
                        b'{"screen_dimensions": {"width_points": "402", "height_points": 874}}'
                    ),
                ),
            ],
        )
        with patcher, self.assertRaises(DeviceError):
            await client.describe()

    async def test_describe_raises_on_unparseable_json(self) -> None:
        client = IDBClient(udid="DEVICE-1")
        patcher, _captured, _env = _patch_subprocess(
            processes=[_FakeProcess(return_code=0, stdout=b"not-json-at-all")],
        )
        with patcher, self.assertRaises(DeviceError) as ctx:
            await client.describe()
        self.assertIn("unparseable", str(ctx.exception).lower())


class IDBClientUDIDMutabilityTest(unittest.IsolatedAsyncioTestCase):
    """
    ``IOSDevice`` resolves device identifiers lazily; ``set_udid`` must
    update the target so subsequent commands hit the right device.
    """

    async def test_set_udid_updates_subsequent_argv(self) -> None:
        client = IDBClient(udid=None)
        client.set_udid(udid="LATE-RESOLVED")
        patcher, captured, captured_env = _patch_subprocess(
            processes=[_FakeProcess(return_code=0)],
        )
        with patcher:
            await client.tap(x=1, y=1)
        self.assertEqual(captured[0][:2], ["idb", "ui"])
        self.assertNotIn("--udid", captured[0])
        assert captured_env[0] is not None
        self.assertEqual(captured_env[0].get("IDB_UDID"), "LATE-RESOLVED")
