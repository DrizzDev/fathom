SELECT source_id,
       execution_id,
       document,
       created_at
FROM search
WHERE tenant_id = :tenant
  AND conversation_id = :thread
  AND source = 'message'
  AND vector @@ plainto_tsquery('simple', :query)
ORDER BY created_at DESC,
         source_id DESC
LIMIT :limit
