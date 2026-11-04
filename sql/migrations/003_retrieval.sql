-- Migration 003 — retrieval fixes (run once in the Supabase SQL Editor; idempotent).
--
-- 1. Replace the ivfflat indexes with HNSW.
--    Measured on 60 labelled questions (scripts/eval_retrieval.py): through the ivfflat index
--    the RPC found the right chunk in the top 3 for 17% of questions, versus 90% for an exact
--    scan of the same embeddings. ivfflat (lists = 100, built when the table was small,
--    probes = 1 at query time) skipped most of the corpus. HNSW keeps recall close to exact.
-- 2. match_cloud_documents gains `exclude_categories`, so the document agent can search our
--    own documentation without picking up crawled competitor pages (and vice versa), and a
--    `filter_provider` for competitor pages. ef_search is raised so the reranker gets a full
--    candidate list.

BEGIN;

DROP INDEX IF EXISTS idx_cloud_documents_embedding;
CREATE INDEX IF NOT EXISTS idx_cloud_documents_embedding_hnsw
    ON cloud_documents USING hnsw (embedding vector_cosine_ops);

DROP INDEX IF EXISTS idx_feedback_memory_embedding;
CREATE INDEX IF NOT EXISTS idx_feedback_memory_embedding_hnsw
    ON cloud_feedback_memory USING hnsw (query_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_cloud_documents_provider
    ON cloud_documents ((metadata->>'provider'));

DROP FUNCTION IF EXISTS match_cloud_documents(vector, float, int, text);
DROP FUNCTION IF EXISTS match_cloud_documents(vector, float, int, text, text[], text);
CREATE FUNCTION match_cloud_documents (
  query_embedding vector(512),
  match_threshold float,
  match_count int,
  filter_category text DEFAULT NULL,
  exclude_categories text[] DEFAULT NULL,
  filter_provider text DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  service_name varchar,
  category varchar,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
SET search_path = public, extensions
SET hnsw.ef_search = 100
SET hnsw.iterative_scan = relaxed_order  -- pgvector >= 0.8; on an older pgvector delete this line
AS $$
BEGIN
  RETURN QUERY
  SELECT
    cd.id,
    cd.service_name,
    cd.category,
    cd.content,
    cd.metadata,
    1 - (cd.embedding <=> query_embedding) AS similarity
  FROM cloud_documents cd
  WHERE 1 - (cd.embedding <=> query_embedding) > match_threshold
    AND (filter_category IS NULL OR cd.category = filter_category)
    AND (exclude_categories IS NULL OR NOT (cd.category = ANY (exclude_categories)))
    AND (filter_provider IS NULL OR cd.metadata->>'provider' = filter_provider)
  ORDER BY cd.embedding <=> query_embedding ASC
  LIMIT match_count;
END;
$$;

REVOKE EXECUTE ON FUNCTION match_cloud_documents(vector, float, int, text, text[], text) FROM PUBLIC, anon, authenticated;
GRANT  EXECUTE ON FUNCTION match_cloud_documents(vector, float, int, text, text[], text) TO service_role;

COMMIT;
