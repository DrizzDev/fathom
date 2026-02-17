"""
Socket-based signal adapter for remote/API control.
Ideal for headless deployments (Temporal workers, containers) and scalable architecture.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel

from fathom.constants import SignalType
from fathom.interfaces.signal import SignalPort

logger = logging.getLogger(__name__)
console = Console()


class SocketSignal(SignalPort):
    """
    Signal adapter listening on a Unix Domain Socket or TCP Port.
    Decouples control from TTY, enabling API-driven interaction.

    Architecture:
    - Event-Driven: Uses asyncio.Event for O(1) notification of pause signals.
    - Non-blocking: Handler dispatches to a queue for consumption by the Executor.
    - Zero-polling: No sleep loops; tasks are parked by the kernel.
    """

    def __init__(self, socket_path: str = "/tmp/fathom.sock") -> None:
        """
        Initialize socket listener.

        Args:
            socket_path: Path for UDS (Unix) or host:port for TCP.
        """
        self.__socket_path = socket_path
        self.__server: Optional[asyncio.AbstractServer] = None
        self.__injected_context: Optional[str] = None
        self.__pause_requested = False

        # High-performance async primitives for O(1) notification
        self.__pause_event = asyncio.Event()
        self.__command_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()

        # Start server immediately
        self.__setup_server()

    def __setup_server(self) -> None:
        """Starts the async socket server."""
        try:
            # Clean up old socket if exists
            if os.path.exists(self.__socket_path):
                os.remove(self.__socket_path)

            loop = asyncio.get_running_loop()

            # Create task to start server
            loop.create_task(self.__start_serving())

            # Print explicit instructions for the user
            console.print(
                Panel.fit(
                    f"[bold yellow]Socket Control Active[/bold yellow]\n"
                    f"Listening on: [cyan]{self.__socket_path}[/cyan]\n\n"
                    "[bold red]DO NOT TYPE IN THIS TERMINAL.[/bold red]\n"
                    "Open a NEW terminal window to send commands:\n\n"
                    f'  [green]Pause:[/green]   echo \'{{"cmd": "pause"}}\' | nc -U {self.__socket_path}\n'
                    f'  [green]Resume:[/green]  echo \'{{"cmd": "resume"}}\' | nc -U {self.__socket_path}\n'
                    f'  [green]Inject:[/green]  echo \'{{"cmd": "inject", "data": "..."}}\' | nc -U {self.__socket_path}',
                    title="Remote Control Instructions",
                    border_style="red",
                )
            )
            logger.info(f"Signal Server listening on {self.__socket_path}")

        except Exception as e:
            logger.error(f"Failed to start signal server: {e}")

    async def __start_serving(self) -> None:
        """Internal server starter."""
        self.__server = await asyncio.start_unix_server(
            self.__handle_client, path=self.__socket_path
        )
        async with self.__server:
            await self.__server.serve_forever()

    async def __handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """
        Handles individual client connections.
        Expects line-delimited JSON.
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

        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            writer.close()

    async def __process_payload(
        self, payload: Dict[str, Any], writer: asyncio.StreamWriter
    ) -> None:
        """Dispatches commands with zero-latency notifications."""
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
                self.__injected_context = content
                await self.__command_queue.put(payload)
                console.print(f"[bold cyan]💡 Remote Context Injected:[/bold cyan] {content}")
                writer.write(b'{"status": "injected"}\n')

        elif cmd == "answer":
            await self.__command_queue.put(payload)
            writer.write(b'{"status": "answer_received"}\n')

        await writer.drain()

    # --- SignalPort Implementation ---

    async def check_signal(self) -> Optional[str]:
        """Non-blocking check for an active pause signal."""
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
        """Blocks until resume/inject command received."""
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
                raise KeyboardInterrupt("Remote cancellation")

    def get_injected_context(self) -> Optional[str]:
        """Atomic retrieval and consumption of context."""
        context = self.__injected_context
        self.__injected_context = None
        return context

    async def ask(self, *, prompt: str) -> str:
        """Requests human input via the remote control channel."""
        logger.info(f"Agent Request: {prompt}")

        while True:
            cmd_data = await self.__command_queue.get()
            if cmd_data.get("cmd") == "answer":
                return cmd_data.get("data", "")
