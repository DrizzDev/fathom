from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from rich.console import Console
from rich.panel import Panel

from fathom.constants import SignalType
from fathom.core.exceptions import WorkflowCancelledError
from fathom.interfaces.signal import SignalPort

console = Console()
logger = logging.getLogger(__name__)


class SocketSignal(SignalPort):
    """
    Signal adapter listening on a Unix Domain Socket or TCP Port.
    Decouples control from TTY, enabling API-driven interaction.
    """

    def __init__(self, socket_path: str = "/tmp/fathom.sock") -> None:  # nosec B108
        """
        Bind the control endpoint (a Unix socket path or ``host:port``) and start the async socket server.
        """

        self.__socket_path = socket_path
        self.__server: Optional[asyncio.AbstractServer] = None

        self.__pause_requested = False
        self.__injected_contexts: Deque[str] = deque()

        # High-performance async primitives for O(1) notification
        self.__pause_event = asyncio.Event()
        self.__command_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        # Start server immediately
        self.__setup_server()

    def supports_interruption(self) -> bool:
        """
        Return interruption support for this adapter.
        """

        return True

    def __setup_server(self) -> None:
        """
        Starts the async socket server.
        """

        try:
            # Clean up old socket if exists
            socket_file = Path(self.__socket_path)
            if socket_file.exists():
                socket_file.unlink()

            loop = asyncio.get_running_loop()

            # Create task to start server
            loop.create_task(self.__start_serving())

            # Print explicit instructions for the user
            console.print(
                Panel.fit(
                    f"[bold yellow]Socket Control Active[/bold yellow]\n"
                    f"Listening on: [cyan]{self.__socket_path}[/cyan]\n\n"
                    "[bold red]DO NOT TYPE IN CURRENT TERMINAL.[/bold red]\n"
                    "Open a NEW terminal window to send commands:\n\n"
                    f'  [green]Pause:[/green]   echo \'{{"cmd": "pause"}}\' | nc -U {self.__socket_path}\n'
                    f'  [green]Resume:[/green]  echo \'{{"cmd": "resume"}}\' | nc -U {self.__socket_path}\n'
                    f'  [green]Inject:[/green]  echo \'{{"cmd": "inject", "data": "..."}}\' | nc -U {self.__socket_path}',
                    title="Remote Control Instructions",
                    border_style="red",
                )
            )
            logger.info(f"Signal Server listening on {self.__socket_path}")

        except Exception as exception:
            logger.error(f"Failed to start signal server: {exception}")

    async def __start_serving(self) -> None:
        """
        Bind the Unix socket server and serve until it is closed.
        """

        self.__server = await asyncio.start_unix_server(
            self.__handle_client,
            path=self.__socket_path,
        )
        async with self.__server:
            await self.__server.serve_forever()

    async def __handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """
        Handles individual client connections. Expects line-delimited JSON.
        """

        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break

                try:
                    payload = json.loads(line.decode().strip())
                    await self.__process_payload(payload, writer)
                except json.JSONDecodeError:
                    writer.write(b'{"error": "Invalid JSON"}\n')
                    await writer.drain()

        except Exception as exception:
            logger.error(f"Client connection error: {exception}")
        finally:
            writer.close()

    async def __process_payload(
        self, payload: Dict[str, Any], writer: asyncio.StreamWriter
    ) -> None:
        """
        Dispatch one control command (pause/resume/inject/answer/cancel) and reply with a status line.
        """

        cmd = payload.get("cmd", "").lower()

        if cmd == "pause":
            self.__pause_requested = True
            # Trigger the event loop to wake up the waiting Executor immediately
            self.__pause_event.set()

            logger.info("Remote Pause Requested")
            console.print("[bold yellow]⏸️  Remote Pause Signal Received[/bold yellow]")
            writer.write(b'{"status": "paused_ack"}\n')

        elif cmd == "resume":
            await self.__command_queue.put(payload)
            # Reset pause state for next run
            self.__pause_requested = False
            self.__pause_event.clear()

            console.print("[bold green]▶️  Remote Resume Signal Received[/bold green]")
            writer.write(b'{"status": "resuming"}\n')

        elif cmd == "inject":
            content = payload.get("data", "")
            if content:
                self.__injected_contexts.append(content)
                await self.__command_queue.put(payload)
                console.print(f"[bold cyan]💡 Remote Context Injected:[/bold cyan] {content}")
                writer.write(b'{"status": "injected"}\n')

        elif cmd == "answer":
            await self.__command_queue.put(payload)
            writer.write(b'{"status": "answer_received"}\n')

        elif cmd == "cancel":
            self.__pause_requested = False
            self.__pause_event.clear()
            await self.__command_queue.put(payload)
            console.print("[bold red]⏹️  Remote Cancel Signal Received[/bold red]")
            writer.write(b'{"status": "cancelled"}\n')

        await writer.drain()

    async def check_signal(self) -> Optional[str]:
        """
        Non-blocking check for an active pause signal.
        """

        return SignalType.ASK.value if self.__pause_requested else None

    async def wait_for_pause(self) -> None:
        """
        Efficiently blocks until a pause signal is received via socket.
        Zero CPU usage - task is parked until __pause_event.set() is called.
        """

        if self.__pause_requested:
            return

        # O(1) wait on event primitive
        await self.__pause_event.wait()

    async def wait_for_resume(self) -> None:
        """
        Blocks until resume/inject command received.
        """

        logger.info("Waiting for remote command (RESUME/INJECT/CANCEL)...")

        while True:
            # Park task until socket handler pushes to queue
            cmd_data = await self.__command_queue.get()
            cmd = cmd_data.get("cmd")

            if cmd == "resume":
                # State reset handled in __process_payload
                break

            elif cmd == "inject":
                # State already updated in __injected_context
                pass

            elif cmd == "cancel":
                self.__pause_requested = False
                self.__pause_event.clear()
                raise WorkflowCancelledError(workflow_id="socket")

    async def get_injected_context(self) -> Optional[str]:
        """
        DEPRECATED: Use peek_next_context and consume_context.
        Atomic retrieval and consumption of context.
        """

        context = self.__injected_contexts[0] if self.__injected_contexts else None
        if self.__injected_contexts:
            self.__injected_contexts.popleft()
        return context

    async def peek_next_context(self) -> Optional[str]:
        """
        Peek at the current injected context.
        """

        return self.__injected_contexts[0] if self.__injected_contexts else None

    async def consume_context(self) -> None:
        """
        Clear the current injected context.
        """

        if self.__injected_contexts:
            self.__injected_contexts.popleft()

    async def is_pause_requested(self) -> bool:
        """
        Check if pause is requested.
        """

        return self.__pause_requested

    async def has_injected_context(self) -> bool:
        """
        Check if there is injected context available.
        """

        return len(self.__injected_contexts) > 0

    async def ask(self, *, prompt: str) -> str:
        """
        Requests human input via the remote control channel.
        """

        logger.info(f"Agent Request: {prompt}")

        while True:
            cmd_data = await self.__command_queue.get()
            if cmd_data.get("cmd") == "answer":
                return str(cmd_data.get("data", ""))

        return ""
