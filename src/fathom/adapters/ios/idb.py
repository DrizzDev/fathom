"""
Async wrapper around the Facebook ``idb`` CLI.

`idb <https://github.com/facebook/idb>`_ is Meta's iOS development bridge:
an Objective-C++ companion daemon plus a Python CLI that drives both
simulators and physical devices over gRPC. We wrap the CLI rather than
the gRPC API because the CLI is the most stable surface and matches the
``xcrun_simctl`` subprocess pattern already used by ``IOSDevice``.

The client owns no transport state — every call shells out to a fresh
``idb`` invocation. The companion daemon (``idb_companion``) and target
selection are handled by ``idb`` itself when ``--udid`` is passed.

Hierarchy: ``idb ui describe-all --json`` returns Apple's accessibility
tree in JSON, not the XCUIElement-shaped XML expected by
``IOSParser``. ``dump_source`` therefore raises ``DeviceError`` to
match the ``XCRUN_SIMCTL`` backend's behaviour; users wanting full
hierarchy grounding stay on ``XCUITEST`` or ``WEBDRIVER_AGENT``.
JSON-to-XML translation is a separate concern and intentionally
out of scope here.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from fathom.core.exceptions import DeviceError

logger = getLogger(__name__)

DEFAULT_IDB_TIMEOUT_SECONDS = 30.0

# Stderr fingerprint for a stale ``idb_companion`` registration. When
# the companion process dies (sim reboot, sleep, crash) its domain
# socket lingers under ``/tmp/idb/<udid>_companion.sock`` and idb
# reports "Failed to connect to companion ... Connection refused" on
# every subsequent subcommand. ``idb disconnect <udid>`` flushes the
# stale entry so the next call spawns a fresh companion.
_STALE_COMPANION_SIGNATURE = "Failed to connect to companion"
_COMPANION_DISCONNECT_TIMEOUT_SECONDS = 5.0


class IDBClient:
    """
    Thin async wrapper around the ``idb`` CLI.

    Every method shells out via ``asyncio.create_subprocess_exec`` —
    no shared transport, no auth state. ``udid`` selects the target
    when set; without it, ``idb`` falls back to its own default
    target resolution (single connected device or
    ``IDB_COMPANION_HOSTNAME`` env var).
    """

    def __init__(
        self,
        *,
        udid: Optional[str] = None,
        executable_path: str = "idb",
        timeout_seconds: float = DEFAULT_IDB_TIMEOUT_SECONDS,
    ) -> None:
        """
        Bind the client to a target UDID and a timeout per call.
        """

        self.__udid = udid
        self.__executable_path = executable_path
        self.__timeout_seconds = timeout_seconds
        self.__cached_point_dimensions: Optional[Tuple[int, int]] = None

    def set_udid(self, *, udid: str) -> None:
        """
        Update the target UDID after construction.

        ``IOSDevice`` resolves a default booted simulator lazily; once
        that resolution lands, the IDB client must target the same UDID
        so subsequent commands hit the right device.
        """

        if udid != self.__udid:
            self.__cached_point_dimensions = None
        self.__udid = udid

    async def tap(self, *, x: float, y: float) -> None:
        """
        Tap the screen at ``(x, y)`` in screen points (idb's HID
        coordinate space, *not* screenshot pixels). Callers should
        convert pixel coords via ``describe()`` before invoking.
        """

        await self.__run(
            ["ui", "tap", str(round(x)), str(round(y))],
            failure_message="idb tap failed",
        )

    async def swipe(
        self,
        *,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        duration_milliseconds: Optional[int] = None,
    ) -> None:
        """
        Swipe from ``(x1, y1)`` to ``(x2, y2)`` in screen points (idb's
        HID coordinate space, *not* screenshot pixels).
        ``duration_milliseconds`` controls the gesture length; ``idb``
        expects seconds via ``--duration``.
        """

        argv = [
            "ui",
            "swipe",
            str(round(x1)),
            str(round(y1)),
            str(round(x2)),
            str(round(y2)),
        ]
        if duration_milliseconds is not None and duration_milliseconds > 0:
            argv += ["--duration", f"{duration_milliseconds / 1000.0:.3f}"]

        await self.__run(argv, failure_message="idb swipe failed")

    async def type_text(self, *, text: str) -> None:
        """
        Type ``text`` into the focused element.
        """

        await self.__run(
            ["ui", "text", text],
            failure_message="idb type failed",
        )

    async def press_home(self) -> None:
        """
        Press the hardware home button (or its software equivalent).
        """

        await self.__run(
            ["ui", "button", "HOME"],
            failure_message="idb home failed",
        )

    async def capture_screen(self) -> bytes:
        """
        Capture a screenshot.

        ``idb screenshot`` writes to a file path (no stdout option), so
        we route through a tempfile and read it back. The tempfile is
        always unlinked, even on subprocess failure.
        """

        # Use mkstemp so we own the path before idb writes; keep the fd
        # closed so idb can open the path for writing on every platform.
        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="fathom-idb-")
        os.close(fd)

        try:
            await self.__run(
                ["screenshot", tmp_path],
                failure_message="idb screenshot failed",
            )
            try:
                with Path(tmp_path).open("rb") as handle:
                    payload = handle.read()
            except OSError as exception:
                raise DeviceError(
                    f"idb screenshot succeeded but tempfile {tmp_path} was unreadable: {exception}"
                ) from exception
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                logger.debug("Could not unlink idb screenshot tempfile: %s", tmp_path)

        if not payload:
            raise DeviceError("idb screenshot wrote an empty file")

        return payload

    async def dump_source(self) -> str:
        """
        Hierarchy extraction is intentionally not supported on the IDB
        backend (see module docstring). Callers should fall back to
        screenshot-only grounding or switch to XCUITEST/WEBDRIVER_AGENT.
        """

        raise DeviceError(
            "idb backend does not expose XCUIElement-shaped XML hierarchy. "
            "Switch to XCUITEST or WEBDRIVER_AGENT for hierarchy grounding."
        )

    async def describe(self) -> Tuple[int, int]:
        """
        Resolve screen size in logical points for pixel→point conversion.

        idb's ``ui tap`` / ``ui swipe`` operate in the HID point
        coordinate space used by UIKit (e.g. 402×874 on an iPhone 17
        simulator), not the pixel grid of ``idb screenshot``'s output
        (e.g. 1206×2622 at density 3×). Callers converting screenshot
        pixel coordinates must scale by the ratio returned here.

        Parses the ``screen_dimensions`` block of ``idb describe --json``
        and caches the result — screen geometry is stable for the life
        of a companion/target binding. ``set_udid`` invalidates the
        cache so a retargeted client re-describes.
        """

        if self.__cached_point_dimensions is not None:
            return self.__cached_point_dimensions

        _return_code, stdout, _stderr = await self.__run(
            ["describe", "--json"],
            failure_message="idb describe failed",
            capture_stdout=True,
        )

        try:
            payload = json.loads(stdout.decode("utf-8", errors="ignore"))
        except ValueError as exception:
            raise DeviceError(f"idb describe returned unparseable JSON: {exception}") from exception

        screen_dimensions = payload.get("screen_dimensions") if isinstance(payload, dict) else None
        if not isinstance(screen_dimensions, dict):
            raise DeviceError("idb describe response missing screen_dimensions block")

        width_points = screen_dimensions.get("width_points")
        height_points = screen_dimensions.get("height_points")
        if not isinstance(width_points, int) or not isinstance(height_points, int):
            raise DeviceError("idb describe returned non-integer point dimensions")
        if width_points <= 0 or height_points <= 0:
            raise DeviceError("idb describe returned non-positive point dimensions")

        self.__cached_point_dimensions = (width_points, height_points)
        return self.__cached_point_dimensions

    async def launch(self, *, bundle_identifier: str) -> None:
        """
        Launch ``bundle_identifier`` on the target.
        """

        await self.__run(
            ["launch", bundle_identifier],
            failure_message=f"idb launch {bundle_identifier} failed",
        )

    async def terminate(self, *, bundle_identifier: str) -> None:
        """
        Terminate ``bundle_identifier`` if it is running.
        """

        await self.__run(
            ["terminate", bundle_identifier],
            failure_message=f"idb terminate {bundle_identifier} failed",
            raise_on_error=False,
        )

    async def list_apps(self) -> List[str]:
        """
        Return installed bundle identifiers on the target.

        Parses ``idb list-apps --json`` output (a JSON list of app
        records keyed by ``bundle_id``). Returns ``[]`` on any parse or
        invocation failure so the caller can fall back to free-text
        entry rather than crashing the wizard.
        """

        return_code, stdout, _ = await self.__run(
            ["list-apps", "--json"],
            failure_message="idb list-apps failed",
            capture_stdout=True,
            raise_on_error=False,
        )
        if return_code != 0 or not stdout:
            return []

        bundles: List[str] = []
        # ``idb list-apps --json`` emits one JSON object per line
        # (NDJSON). Parse defensively — any malformed line is dropped.
        for line in stdout.decode("utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except ValueError:
                continue
            bundle_id = payload.get("bundle_id") if isinstance(payload, dict) else None
            if isinstance(bundle_id, str) and bundle_id:
                bundles.append(bundle_id)

        return sorted(set(bundles))

    @classmethod
    def is_available(cls, *, executable_path: str = "idb") -> bool:
        """
        Whether the ``idb`` CLI is on ``PATH``. Best-effort, never raises.
        """

        return shutil.which(executable_path) is not None

    async def __run(
        self,
        argv: Sequence[str],
        *,
        failure_message: str,
        capture_stdout: bool = False,
        capture_stderr: bool = True,
        raise_on_error: bool = True,
        allow_companion_recovery: bool = True,
    ) -> Tuple[int, bytes, bytes]:
        """
        Invoke ``idb`` with the given argv tail.

        ``--udid`` is a per-subcommand flag in idb's argparse (not a
        root flag), and its position varies between leaf subcommands
        (e.g. ``ui tap`` vs ``screenshot``). To avoid argv surgery for
        every command, we forward the target via the ``IDB_UDID`` env
        var — explicitly supported by idb and applied uniformly.

        ``raise_on_error=False`` is for opportunistic calls (terminate,
        list-apps) that should degrade rather than blow up.

        ``allow_companion_recovery`` enables a one-shot retry when the
        failure matches the stale-companion fingerprint — the retry
        bypasses recovery to prevent infinite loops.
        """

        if not argv:
            raise DeviceError("IDBClient.__run requires at least a subcommand in argv")

        return_code, stdout, stderr = await self.__invoke(
            argv=argv,
            failure_message=failure_message,
            capture_stdout=capture_stdout,
            capture_stderr=capture_stderr,
        )

        # One-shot stale-companion recovery. idb caches companion
        # registrations in ``/tmp/idb/state``; when the backing process
        # dies the entry goes dangling and every subcommand fails with
        # a "Failed to connect to companion" banner until something
        # runs ``idb disconnect <udid>``. We handle that transparently
        # so the agent survives sim reboots without operator
        # intervention.
        if (
            return_code != 0
            and allow_companion_recovery
            and self.__is_stale_companion_failure(stderr=stderr)
        ):
            await self.__recover_stale_companion()
            return await self.__run(
                argv,
                failure_message=failure_message,
                capture_stdout=capture_stdout,
                capture_stderr=capture_stderr,
                raise_on_error=raise_on_error,
                allow_companion_recovery=False,
            )

        if return_code != 0 and raise_on_error:
            stderr_text = (stderr or b"").decode("utf-8", errors="ignore").strip()
            raise DeviceError(
                f"{failure_message} (exit={return_code}): {stderr_text or 'no stderr'}"
            )

        return return_code, stdout or b"", stderr or b""

    async def __invoke(
        self,
        *,
        argv: Sequence[str],
        failure_message: str,
        capture_stdout: bool,
        capture_stderr: bool,
    ) -> Tuple[int, bytes, bytes]:
        """
        Spawn a single ``idb`` subprocess and return ``(return_code,
        stdout, stderr)``. Raises ``DeviceError`` only for conditions
        that cannot be recovered by retrying (missing binary, timeout).
        """

        full_argv: List[str] = [self.__executable_path, *argv]

        env: Optional[Dict[str, str]] = None
        if self.__udid:
            env = dict(os.environ)
            env["IDB_UDID"] = self.__udid

        try:
            process = await asyncio.create_subprocess_exec(
                *full_argv,
                stdout=asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE if capture_stderr else asyncio.subprocess.DEVNULL,
                env=env,
            )
        except FileNotFoundError as exception:
            raise DeviceError(
                f"idb CLI not found at {self.__executable_path!r}; install via "
                "`brew tap facebook/fb && brew install idb-companion && pip install fb-idb`"
            ) from exception

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.__timeout_seconds
            )
        except asyncio.TimeoutError as exception:
            process.kill()
            await process.wait()
            raise DeviceError(
                f"{failure_message}: timed out after {self.__timeout_seconds:.1f}s"
            ) from exception

        return_code = process.returncode if process.returncode is not None else -1
        return return_code, stdout or b"", stderr or b""

    @staticmethod
    def __is_stale_companion_failure(*, stderr: bytes) -> bool:
        """
        Detect the stderr fingerprint idb prints when its cached
        companion registration points at a dead process.
        """

        return _STALE_COMPANION_SIGNATURE in stderr.decode("utf-8", errors="ignore")

    async def __recover_stale_companion(self) -> None:
        """
        Best-effort ``idb disconnect <udid>`` to flush the stale
        companion entry. Errors are swallowed — if recovery itself
        fails the follow-up retry will surface the underlying idb
        error, which is more informative than a recovery failure.
        """

        if not self.__udid:
            return

        try:
            process = await asyncio.create_subprocess_exec(
                self.__executable_path,
                "disconnect",
                self.__udid,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(
                process.wait(),
                timeout=_COMPANION_DISCONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, asyncio.TimeoutError) as exception:
            logger.debug("idb companion recovery failed: %s", exception)
            return
