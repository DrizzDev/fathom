SELECT id
FROM conversations
WHERE tenant_id = :tenant
  AND id = :thread
  AND deleted_at IS NULL
