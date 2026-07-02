UPDATE conversations
SET updated_at = GREATEST(updated_at, :updated),
    digest = :digest
WHERE tenant_id = :tenant
  AND id = :thread
  AND deleted_at IS NULL
  AND NOT EXISTS (
      SELECT 1
      FROM events
      WHERE events.tenant_id = :tenant
        AND events.conversation_id = :thread
        AND events.sequence > :sequence
  )
