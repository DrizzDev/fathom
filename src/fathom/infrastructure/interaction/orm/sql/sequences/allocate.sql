INSERT INTO sequences (
    id,
    tenant_id,
    conversation_id,
    scope,
    value,
    created_at
)
VALUES (:id, :tenant, :thread, :scope, 1, now())
ON CONFLICT (tenant_id, conversation_id, scope)
DO UPDATE SET
    value = sequences.value + 1
RETURNING value
