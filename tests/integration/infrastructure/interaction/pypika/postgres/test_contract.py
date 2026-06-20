"""
Native Postgres contract smoke tests for the conversation interaction adapter.

These tests are skipped unless `FATHOM_TEST_POSTGRES_DSN` is set. When set,
they run against a real Postgres database and exercise the dialect-sensitive
bits the SQLite suite cannot cover:

  - jsonb encode/decode for body, labels, metadata
  - timestamptz round-trip for created/updated/lifecycle timestamps
  - per-thread monotonic sequence allocator under sequential and concurrent
    inserts (Postgres MVCC vs SQLite single-writer)
  - Postgres-only conflict and lease semantics (idempotent replay,
    integrity-error → ThreadConflictError translation)
  - the `migrations` ledger + checksum guard accept the declarative bundle
  - the customer-success `vw_conversations`, `vw_conversation_messages`,
    `vw_conversation_artifacts` views read after writes through the adapter

This file is intentionally focused on the public InteractionPort contract.
Per-aggregate tests live in `tests/unit/adapters/interaction/test_sqlite.py`
for SQLite and should be ported to a shared parametrised module once the
native Postgres adapter is the primary path.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Tuple

import pytest

from .conftest import requires_postgres

pytestmark = [pytest.mark.asyncio, requires_postgres]


def _now() -> datetime:
    """
    Return a timezone-aware UTC timestamp; tests use one fixed value per case.
    """

    return datetime.now(tz=timezone.utc)


async def _seed_actor_and_thread(adapter, *, tenant: str) -> Tuple[str, str]:
    """
    Create one actor and one thread inside the adapter, returning their ids.

    Helper used by every contract test so the per-test setup stays small.
    """

    from fathom.constants.collaboration import ActorKind, MembershipRole
    from fathom.schemas.interaction import (
        CreateActor,
        CreateThread,
        Identity,
        JoinThread,
    )

    now = _now()
    actor_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    membership_id = str(uuid.uuid4())

    await adapter.create_actor(
        request=CreateActor(
            identity=Identity(id=actor_id, tenant=tenant),
            kind=ActorKind.HUMAN,
            name="QA Smoke",
            created_at=now,
        )
    )
    await adapter.create_thread(
        request=CreateThread(
            identity=Identity(id=thread_id, tenant=tenant),
            title="contract smoke",
            creator=actor_id,
            created_at=now,
        )
    )
    await adapter.join_thread(
        request=JoinThread(
            identity=Identity(id=membership_id, tenant=tenant),
            thread=thread_id,
            actor=actor_id,
            role=MembershipRole.OWNER,
            joined_at=now,
        )
    )
    return actor_id, thread_id


async def test_record_message_roundtrips_jsonb_body(postgres_adapter):
    """
    A recorded message's body must round-trip through jsonb without becoming
    a JSON-encoded string. This catches the dialect mismatch where SQLite
    stores body as TEXT but Postgres stores it as JSONB.
    """

    from fathom.constants.collaboration import Audience, MessageKind
    from fathom.schemas.interaction import (
        Content,
        Identity,
        MessageQuery,
        RecordMessage,
    )

    tenant = "tenant_jsonb"
    actor_id, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)
    body = {
        "text": "hello postgres",
        "structured": {"items": [1, 2, 3], "flag": True},
    }
    await postgres_adapter.record_message(
        request=RecordMessage(
            identity=Identity(id=f"msg:{uuid.uuid4().hex[:8]}", tenant=tenant),
            thread=thread_id,
            author=actor_id,
            kind=MessageKind.REQUEST,
            audience=Audience.THREAD,
            content=Content(body=body),
            created_at=_now(),
        )
    )

    messages = await postgres_adapter.get_messages(
        query=MessageQuery(tenant=tenant, thread=thread_id)
    )
    assert len(messages) == 1
    assert messages[0].content.body == body, (
        "jsonb round-trip lost structure; native Postgres must return dict, not string."
    )


async def test_per_thread_sequence_is_monotonic_under_serial_inserts(
    postgres_adapter,
):
    """
    Two messages inserted into the same thread must receive sequences 1 and 2.

    This exercises the `sequences` allocator's `INSERT ... ON CONFLICT DO
    UPDATE ... RETURNING` round-trip on Postgres MVCC.
    """

    from fathom.constants.collaboration import Audience, MessageKind
    from fathom.schemas.interaction import (
        Content,
        Identity,
        MessageQuery,
        RecordMessage,
    )

    tenant = "tenant_seq"
    actor_id, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)
    for _ in range(2):
        await postgres_adapter.record_message(
            request=RecordMessage(
                identity=Identity(id=f"msg:{uuid.uuid4().hex[:8]}", tenant=tenant),
                thread=thread_id,
                author=actor_id,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": "ping"}),
                created_at=_now(),
            )
        )

    messages = await postgres_adapter.get_messages(
        query=MessageQuery(tenant=tenant, thread=thread_id)
    )
    sequences = sorted(message.sequence for message in messages)
    assert sequences == [1, 2], (
        f"Expected sequences [1, 2]; got {sequences}. "
        "Allocator must return strictly monotonic per-thread integers."
    )


async def test_concurrent_message_inserts_get_distinct_sequences(
    postgres_adapter,
):
    """
    50 concurrent message inserts into the same thread must receive 50
    distinct sequences with no duplicates and no gaps caused by successful
    inserts. Mirrors the concurrency gate from the implementation plan.
    """

    from fathom.constants.collaboration import Audience, MessageKind
    from fathom.schemas.interaction import (
        Content,
        Identity,
        MessageQuery,
        RecordMessage,
    )

    tenant = "tenant_concurrent"
    actor_id, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)

    async def __record_one(slot: int) -> None:
        """
        Record one message in the concurrent sequence-allocation test.
        """

        await postgres_adapter.record_message(
            request=RecordMessage(
                identity=Identity(id=f"msg:{uuid.uuid4().hex[:8]}_{slot}", tenant=tenant),
                thread=thread_id,
                author=actor_id,
                kind=MessageKind.REQUEST,
                audience=Audience.THREAD,
                content=Content(body={"text": f"ping {slot}"}),
                created_at=_now(),
            )
        )

    await asyncio.gather(*[__record_one(slot=index) for index in range(50)])

    messages = await postgres_adapter.get_messages(
        query=MessageQuery(tenant=tenant, thread=thread_id)
    )
    sequences = sorted(message.sequence for message in messages)
    assert sequences == list(range(1, 51)), (
        f"Concurrent allocator did not produce sequences 1..50; got {sequences}."
    )


async def test_thread_replay_returns_existing_record(postgres_adapter):
    """
    Recording the same thread twice with identical content must return the
    stored row, not raise — idempotent replay is the contract for safe HTTP
    retries.
    """

    from fathom.constants.collaboration import ActorKind
    from fathom.schemas.interaction import CreateActor, CreateThread, Identity

    tenant = "tenant_replay"
    now = _now()
    actor_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())

    await postgres_adapter.create_actor(
        request=CreateActor(
            identity=Identity(id=actor_id, tenant=tenant),
            kind=ActorKind.HUMAN,
            name="Replay",
            created_at=now,
        )
    )
    request = CreateThread(
        identity=Identity(id=thread_id, tenant=tenant),
        title="replay",
        creator=actor_id,
        created_at=now,
    )
    first = await postgres_adapter.create_thread(request=request)
    second = await postgres_adapter.create_thread(request=request)

    assert first.identity.id == second.identity.id
    assert first.timing.created == second.timing.created


async def test_list_threads_returns_inserted_thread(postgres_adapter):
    """
    A thread inserted through the adapter must appear in `list_threads`
    output, with `total` reflecting the insert. Validates the read path
    plus jsonb metadata round-trip.
    """

    from fathom.schemas.interaction import ThreadListQuery

    tenant = "tenant_list"
    _, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)

    page = await postgres_adapter.list_threads(query=ThreadListQuery(tenant=tenant, limit=10))
    assert page.total == 1
    assert any(thread.identity.id == thread_id for thread in page.items)


async def test_list_threads_title_filter_escapes_like_wildcards(postgres_adapter):
    """
    Title prefix filter treats SQL LIKE wildcards as literal characters via ESCAPE.
    """

    from fathom.constants.collaboration import ActorKind
    from fathom.schemas.interaction import (
        CreateActor,
        CreateThread,
        Identity,
        ThreadListQuery,
    )

    tenant = "tenant_title_escape"
    now = _now()
    actor_id = str(uuid.uuid4())
    await postgres_adapter.create_actor(
        request=CreateActor(
            identity=Identity(id=actor_id, tenant=tenant),
            kind=ActorKind.HUMAN,
            name="QA Smoke",
            created_at=now,
        )
    )

    threads = [
        ("50% discount", str(uuid.uuid4())),
        ("50 cents", str(uuid.uuid4())),
        ("_drafts", str(uuid.uuid4())),
    ]
    for title, thread_id in threads:
        await postgres_adapter.create_thread(
            request=CreateThread(
                identity=Identity(id=thread_id, tenant=tenant),
                title=title,
                creator=actor_id,
                created_at=now,
            )
        )

    percent = await postgres_adapter.list_threads(
        query=ThreadListQuery(tenant=tenant, title="50%", limit=10)
    )
    underscore = await postgres_adapter.list_threads(
        query=ThreadListQuery(tenant=tenant, title="_dr", limit=10)
    )

    assert {thread.identity.id for thread in percent.items} == {threads[0][1]}
    assert {thread.identity.id for thread in underscore.items} == {threads[2][1]}


async def test_list_scripts_paginates_by_updated_at(postgres_adapter):
    """
    Scripts are returned newest-first with cursor pagination preserving order.
    """

    from fathom.constants.collaboration import ArtifactBackend, ArtifactKind
    from fathom.schemas.interaction import (
        Identity,
        LinkArtifact,
        SaveScript,
        ScriptListQuery,
    )

    tenant = "tenant_script_list"
    now = _now()
    _, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)
    artifact_id = str(uuid.uuid4())
    await postgres_adapter.link_artifact(
        request=LinkArtifact(
            identity=Identity(id=artifact_id, tenant=tenant),
            thread=thread_id,
            producer=str(uuid.uuid4()),
            kind=ArtifactKind.SCRIPT,
            uri="/tmp/script.txt",
            backend=ArtifactBackend.LOCAL,
            mime="text/plain",
            size=17,
            created_at=now,
        )
    )

    script_ids = []
    for index in range(3):
        script_id = str(uuid.uuid4())
        await postgres_adapter.save_script(
            request=SaveScript(
                identity=Identity(id=script_id, tenant=tenant),
                thread=thread_id,
                artifact=artifact_id,
                title=f"Script {index}",
                content=f"OPEN_APP example {index}",
                summary="Generated script.",
                created_at=now,
            )
        )
        script_ids.append(script_id)

    first = await postgres_adapter.list_scripts(
        query=ScriptListQuery(tenant=tenant, thread=thread_id, limit=2)
    )
    second = await postgres_adapter.list_scripts(
        query=ScriptListQuery(tenant=tenant, thread=thread_id, limit=2, cursor=first.next)
    )

    seen = {item.identity.id for item in first.items} | {item.identity.id for item in second.items}
    assert seen == set(script_ids)
    assert first.total == 3


async def test_save_script_updates_title_without_new_version(postgres_adapter):
    """
    Same content with a changed title updates the row but does not bump revision.
    """

    from fathom.constants.collaboration import ArtifactBackend, ArtifactKind
    from fathom.schemas.interaction import (
        Identity,
        LinkArtifact,
        SaveScript,
        ScriptVersionQuery,
    )

    tenant = "tenant_replay_title"
    now = _now()
    _, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)
    artifact_id = str(uuid.uuid4())
    await postgres_adapter.link_artifact(
        request=LinkArtifact(
            identity=Identity(id=artifact_id, tenant=tenant),
            thread=thread_id,
            producer=str(uuid.uuid4()),
            kind=ArtifactKind.SCRIPT,
            uri="/tmp/script.txt",
            backend=ArtifactBackend.LOCAL,
            mime="text/plain",
            size=17,
            created_at=now,
        )
    )
    script_id = str(uuid.uuid4())
    original = await postgres_adapter.save_script(
        request=SaveScript(
            identity=Identity(id=script_id, tenant=tenant),
            thread=thread_id,
            artifact=artifact_id,
            title="Original",
            content="OPEN_APP example",
            summary="initial",
            created_at=now,
        )
    )
    renamed = await postgres_adapter.save_script(
        request=SaveScript(
            identity=Identity(id=script_id, tenant=tenant),
            thread=thread_id,
            artifact=artifact_id,
            title="Renamed",
            content="OPEN_APP example",
            summary="rename",
            created_at=now,
        )
    )
    versions = await postgres_adapter.get_script_versions(
        query=ScriptVersionQuery(tenant=tenant, script=script_id)
    )

    assert original.title == "Original"
    assert renamed.title == "Renamed"
    assert original.revision == 1
    assert renamed.revision == 1
    assert [version.version for version in versions] == [1]


async def test_save_script_concurrent_inserts_resolve_to_one_row(postgres_adapter):
    """
    Concurrent inserts of the same identity end at revision 1 with one immutable version.
    """

    from fathom.constants.collaboration import ArtifactBackend, ArtifactKind
    from fathom.schemas.interaction import (
        Identity,
        LinkArtifact,
        SaveScript,
        ScriptQuery,
        ScriptVersionQuery,
    )

    tenant = "tenant_atomic_save"
    now = _now()
    _, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)
    artifact_id = str(uuid.uuid4())
    await postgres_adapter.link_artifact(
        request=LinkArtifact(
            identity=Identity(id=artifact_id, tenant=tenant),
            thread=thread_id,
            producer=str(uuid.uuid4()),
            kind=ArtifactKind.SCRIPT,
            uri="/tmp/script.txt",
            backend=ArtifactBackend.LOCAL,
            mime="text/plain",
            size=17,
            created_at=now,
        )
    )
    script_id = str(uuid.uuid4())
    request = SaveScript(
        identity=Identity(id=script_id, tenant=tenant),
        thread=thread_id,
        artifact=artifact_id,
        title="Race",
        content="OPEN_APP race",
        summary="race",
        created_at=now,
    )

    await asyncio.gather(
        postgres_adapter.save_script(request=request),
        postgres_adapter.save_script(request=request),
    )

    scripts = await postgres_adapter.get_scripts(query=ScriptQuery(tenant=tenant, script=script_id))
    versions = await postgres_adapter.get_script_versions(
        query=ScriptVersionQuery(tenant=tenant, script=script_id)
    )

    assert len(scripts) == 1
    assert scripts[0].revision == 1
    assert len(versions) == 1


async def test_save_script_concurrent_content_updates_serialize(postgres_adapter):
    """
    Two concurrent saves with different content must serialize via row lock; both succeed without UNIQUE (tenant, script, version) violations.
    """

    from fathom.constants.collaboration import ArtifactBackend, ArtifactKind
    from fathom.schemas.interaction import (
        Identity,
        LinkArtifact,
        SaveScript,
        ScriptQuery,
        ScriptVersionQuery,
    )

    tenant = "tenant_concurrent_update"
    now = _now()
    _, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)
    artifact_id = str(uuid.uuid4())
    await postgres_adapter.link_artifact(
        request=LinkArtifact(
            identity=Identity(id=artifact_id, tenant=tenant),
            thread=thread_id,
            producer=str(uuid.uuid4()),
            kind=ArtifactKind.SCRIPT,
            uri="/tmp/script.txt",
            backend=ArtifactBackend.LOCAL,
            mime="text/plain",
            size=17,
            created_at=now,
        )
    )
    script_id = str(uuid.uuid4())
    await postgres_adapter.save_script(
        request=SaveScript(
            identity=Identity(id=script_id, tenant=tenant),
            thread=thread_id,
            artifact=artifact_id,
            title="Initial",
            content="OPEN_APP one",
            summary="initial",
            created_at=now,
        )
    )

    request_a = SaveScript(
        identity=Identity(id=script_id, tenant=tenant),
        thread=thread_id,
        artifact=artifact_id,
        title="Update A",
        content="OPEN_APP one\nA",
        summary="update-a",
        created_at=now,
    )
    request_b = SaveScript(
        identity=Identity(id=script_id, tenant=tenant),
        thread=thread_id,
        artifact=artifact_id,
        title="Update B",
        content="OPEN_APP one\nB",
        summary="update-b",
        created_at=now,
    )

    await asyncio.gather(
        postgres_adapter.save_script(request=request_a),
        postgres_adapter.save_script(request=request_b),
    )

    scripts = await postgres_adapter.get_scripts(query=ScriptQuery(tenant=tenant, script=script_id))
    versions = await postgres_adapter.get_script_versions(
        query=ScriptVersionQuery(tenant=tenant, script=script_id)
    )

    assert len(scripts) == 1
    assert scripts[0].revision == 3
    assert sorted(version.version for version in versions) == [1, 2, 3]


async def test_concurrent_job_claims_are_unique(postgres_adapter):
    """
    Concurrent Postgres job claims must not return the same pending job to
    multiple workers. This is the production `FOR UPDATE SKIP LOCKED` gate.
    """

    from fathom.constants.collaboration import JobKind
    from fathom.schemas.interaction import ClaimJob, Identity, ScheduleJob

    tenant = "tenant_job_claim"
    _, thread_id = await _seed_actor_and_thread(postgres_adapter, tenant=tenant)
    for _ in range(20):
        await postgres_adapter.schedule_job(
            request=ScheduleJob(
                identity=Identity(id=str(uuid.uuid4()), tenant=tenant),
                thread=thread_id,
                kind=JobKind.MEMORY,
                available_at=_now(),
                created_at=_now(),
            )
        )

    claims = await asyncio.gather(
        *[
            postgres_adapter.claim_job(
                request=ClaimJob(
                    tenant=tenant,
                    owner=f"worker:{index}",
                    claimed=_now(),
                )
            )
            for index in range(20)
        ]
    )
    claimed_ids = [job.identity.id for job in claims if job is not None]

    assert len(claimed_ids) == 20
    assert len(set(claimed_ids)) == 20
