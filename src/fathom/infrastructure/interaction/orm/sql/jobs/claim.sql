UPDATE jobs
SET owner = :owner,
    locked_at = :locked_at,
    state = 'claimed',
    attempts = attempts + 1,
    updated_at = :locked_at
WHERE id = (
    SELECT id
    FROM jobs
    WHERE tenant_id = :tenant
      AND state = 'pending'
      AND available_at <= :available_at
      AND (:job::text IS NULL OR id = :job)
      AND (:kind::text IS NULL OR kind = :kind)
    ORDER BY available_at ASC, id ASC
    LIMIT 1
    FOR UPDATE SKIP LOCKED
)
  AND tenant_id = :tenant
RETURNING id,
          tenant_id AS tenant,
          workspace_id AS workspace,
          conversation_id AS thread,
          execution_id AS execution,
          task_id AS task,
          kind,
          state,
          attempts,
          owner,
          locked_at,
          available_at,
          payload,
          code,
          detail,
          metadata,
          created_at,
          updated_at
