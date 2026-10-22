from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

from fathom.app_graph.graph import AppGraph
from fathom.app_graph.ledger import Ledger
from fathom.app_graph.state import Completion, Description, Identity, Observation, Snapshot, Start
from fathom.infrastructure.apps.parser import Parser
from fathom.interfaces.apps import ApplicationsPort


class Repository(ApplicationsPort):
    """Single-worker transactional graph repository with local evidence files."""

    def __init__(self, *, root: Path) -> None:
        """Bind graph data and evidence to one configured persistence root."""
        self.__root = root
        self.__root.mkdir(parents=True, exist_ok=True)
        with self.__connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS applications (identity TEXT PRIMARY KEY, tenant TEXT NOT NULL, package TEXT NOT NULL, state TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS observations (identity TEXT NOT NULL, execution TEXT NOT NULL, sequence INTEGER NOT NULL, PRIMARY KEY(identity, execution, sequence))"
            )

    @contextmanager
    def __connect(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back and close every repository connection."""
        connection = sqlite3.connect(self.__root / "applications.db", timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    async def start(self, *, start: Start) -> None:
        """Register a persistent Block without blocking the event loop."""
        await asyncio.to_thread(self.__start, start=start)

    def __start(self, *, start: Start) -> None:
        """Serialize idempotent Block registration."""
        with self.__connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            graph = self.__load(connection=connection, identity=start.identity)
            Ledger().start(graph=graph, start=start)
            self.__save(connection=connection, identity=start.identity, graph=graph)

    @staticmethod
    def __key(*, identity: Identity) -> str:
        """Derive a filesystem-safe key without exposing tenant identifiers."""
        return hashlib.sha256(identity.model_dump_json().encode()).hexdigest()

    def __load(self, *, connection: sqlite3.Connection, identity: Identity) -> AppGraph:
        """Load the application or create an empty graph within the transaction."""
        key = self.__key(identity=identity)
        row = connection.execute(
            "SELECT state FROM applications WHERE identity=?", (key,)
        ).fetchone()
        graph = AppGraph(package=identity.package, label=identity.package)
        if row:
            graph.restore(snapshot=Snapshot.model_validate_json(row[0]))
        return graph

    def __save(
        self, *, connection: sqlite3.Connection, identity: Identity, graph: AppGraph
    ) -> None:
        """Publish validated state within the caller's transaction."""
        connection.execute(
            "INSERT INTO applications VALUES(?,?,?,?) ON CONFLICT(identity) DO UPDATE SET state=excluded.state",
            (
                self.__key(identity=identity),
                identity.tenant,
                identity.package,
                graph.snapshot().model_dump_json(),
            ),
        )

    async def observe(self, *, observation: Observation) -> None:
        """Merge discoveries without blocking the event loop."""
        await asyncio.to_thread(self.__observe, observation=observation)

    def __observe(self, *, observation: Observation) -> None:
        """Serialize graph mutations and ignore replayed capture receipts."""
        key = self.__key(identity=observation.identity)
        with self.__connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            receipt = (key, observation.execution, observation.sequence)
            if connection.execute(
                "SELECT 1 FROM observations WHERE identity=? AND execution=? AND sequence=?",
                receipt,
            ).fetchone():
                return
            graph = self.__load(connection=connection, identity=observation.identity)
            stamp = hashlib.sha256(
                f"{observation.execution}:{observation.sequence}".encode()
            ).hexdigest()
            directory = self.__root / key / "evidence"
            directory.mkdir(parents=True, exist_ok=True)
            image = directory / f"{stamp}.png"
            image.write_bytes(observation.image)
            xml = directory / f"{stamp}.xml"
            if observation.xml is not None:
                xml.write_text(observation.xml, encoding="utf-8")
            parsed = Parser().enrich(
                screen=observation.screen, xml=observation.xml, package=observation.identity.package
            )
            screen = parsed.model_copy(
                update={
                    "screenshot_path": str(image),
                    "xml_path": str(xml) if observation.xml is not None else None,
                }
            )
            Ledger().observe(
                graph=graph, observation=observation.model_copy(update={"screen": screen})
            )
            self.__save(connection=connection, identity=observation.identity, graph=graph)
            connection.execute("INSERT INTO observations VALUES(?,?,?)", receipt)

    async def describe(self, *, description: Description) -> None:
        """Persist grounded descriptions without blocking the event loop."""
        await asyncio.to_thread(self.__describe, description=description)

    def __describe(self, *, description: Description) -> None:
        """Serialize description updates with exploration discoveries."""
        with self.__connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            graph = self.__load(connection=connection, identity=description.identity)
            Ledger().describe(graph=graph, description=description)
            self.__save(connection=connection, identity=description.identity, graph=graph)

    async def complete(self, *, completion: Completion) -> None:
        """Finalize a block without blocking the event loop."""
        await asyncio.to_thread(self.__complete, completion=completion)

    def __complete(self, *, completion: Completion) -> None:
        """Serialize completion against pending observations."""
        with self.__connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            graph = self.__load(connection=connection, identity=completion.identity)
            Ledger().complete(graph=graph, completion=completion)
            self.__save(connection=connection, identity=completion.identity, graph=graph)

    async def read(self, *, identity: Identity) -> Snapshot:
        """Read a committed graph snapshot."""
        return await asyncio.to_thread(self.__read, identity=identity)

    def __read(self, *, identity: Identity) -> Snapshot:
        """Return the current transactionally consistent application state."""
        with self.__connect() as connection:
            return self.__load(connection=connection, identity=identity).snapshot()

    async def catalog(self, *, tenant: str) -> list[Snapshot]:
        """List only application graphs in the trusted tenant."""
        return await asyncio.to_thread(self.__catalog, tenant=tenant)

    def __catalog(self, *, tenant: str) -> list[Snapshot]:
        """Read committed snapshots for one organization."""
        with self.__connect() as connection:
            return [
                Snapshot.model_validate_json(row[0])
                for row in connection.execute(
                    "SELECT state FROM applications WHERE tenant=? ORDER BY package", (tenant,)
                )
            ]
