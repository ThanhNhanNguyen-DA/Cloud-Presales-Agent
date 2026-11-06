-- Additive migration, after 003. No document rows are deleted or re-embedded.
-- Multilingual "simple" preserves product names and abbreviations. Vietnamese
-- diacritics are preserved; semantic search remains the complementary channel.
BEGIN;

ALTER TABLE public.cloud_documents ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
    setweight(to_tsvector('simple'::regconfig, coalesce(service_name, '')), 'A') ||
    setweight(to_tsvector('simple'::regconfig, coalesce(category, '')), 'B') ||
    setweight(to_tsvector('simple'::regconfig, content), 'C')
) STORED;

CREATE INDEX IF NOT EXISTS idx_cloud_documents_search_vector
    ON public.cloud_documents USING gin (search_vector);

CREATE OR REPLACE FUNCTION public.search_cloud_documents_lexical (
    query_text text,
    match_count int DEFAULT 15,
    filter_category text DEFAULT NULL,
    exclude_categories text[] DEFAULT NULL,
    filter_provider text DEFAULT NULL
)
RETURNS TABLE (
    id uuid, service_name varchar, category varchar, content text,
    metadata jsonb, lexical_score real
)
LANGUAGE sql STABLE
SET search_path = public, extensions
AS $$
    SELECT cd.id, cd.service_name, cd.category, cd.content, cd.metadata,
           ts_rank_cd(cd.search_vector, websearch_to_tsquery('simple', query_text)) AS lexical_score
    FROM public.cloud_documents cd
    WHERE cd.search_vector @@ websearch_to_tsquery('simple', query_text)
      AND (filter_category IS NULL OR cd.category = filter_category)
      AND (exclude_categories IS NULL OR cd.category IS NULL OR NOT (cd.category = ANY(exclude_categories)))
      AND (filter_provider IS NULL OR cd.metadata->>'provider' = filter_provider)
      AND coalesce(cd.metadata->>'source', '') NOT IN ('', 'sample_data')
      AND coalesce(cd.metadata->>'review_status', '') NOT IN ('pending', 'rejected')
    ORDER BY lexical_score DESC, cd.id
    LIMIT greatest(1, least(match_count, 100));
$$;

REVOKE EXECUTE ON FUNCTION public.search_cloud_documents_lexical(text, int, text, text[], text)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.search_cloud_documents_lexical(text, int, text, text[], text)
    TO service_role;

COMMIT;
