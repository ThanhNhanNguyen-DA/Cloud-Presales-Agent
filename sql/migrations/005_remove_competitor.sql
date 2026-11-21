-- Migration 005 — remove the Competitor Agent's data (run after 004; idempotent).
-- The competitor specialist was removed from the app. This deletes its data:
--   * crawled competitor pages in cloud_documents (category 'Competitor');
--   * the AWS/Azure/GCP mapping table;
--   * the provider index, used only to filter competitor pages.
-- Earlier migrations are left unchanged so a fresh install (schema.sql, then 002..005) still works.
-- The match_cloud_documents / search_cloud_documents_lexical RPCs keep their optional
-- filter_provider argument (unused, defaults to NULL).

BEGIN;

-- Not metadata->>'type' = 'competitor_crawl': our own CMC product pages (role 'own') were
-- crawled by the same script and carry that type too.
DELETE FROM public.cloud_documents
WHERE category = 'Competitor'
   OR metadata->>'role' = 'competitor';

DROP TABLE IF EXISTS public.cloud_competitor_mapping;

DROP INDEX IF EXISTS public.idx_cloud_documents_provider;

COMMIT;
