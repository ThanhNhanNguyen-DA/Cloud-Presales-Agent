-- Migration 002 — security hardening (run once in the Supabase SQL Editor).
-- Safe to re-run: every statement is idempotent.
--
-- 1. Feedback review workflow: a human correction is only used after an admin approves it.
-- 2. Row Level Security on every table, with NO policies: the anon / authenticated roles
--    (i.e. anyone holding the public anon key) can no longer read or write anything.
--    The backend uses a secret / service_role key, which bypasses RLS, so it keeps working.
-- 3. The RPC functions can no longer be called with the anon key.

BEGIN;

-- ---------------------------------------------------------------------------------------
-- 1. Feedback review columns
-- ---------------------------------------------------------------------------------------
ALTER TABLE cloud_feedback_memory
    ADD COLUMN IF NOT EXISTS review_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS submitted_by TEXT,
    ADD COLUMN IF NOT EXISTS reviewed_by TEXT,
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP WITH TIME ZONE;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'cloud_feedback_memory_review_status_check'
    ) THEN
        ALTER TABLE cloud_feedback_memory
            ADD CONSTRAINT cloud_feedback_memory_review_status_check
            CHECK (review_status IN ('pending', 'approved', 'rejected'));
    END IF;
END $$;

-- Rows without a correction never reach a prompt, so they need no review.
-- Existing corrections stay 'pending' until someone reviews them via /admin/feedback.
UPDATE cloud_feedback_memory
SET review_status = 'approved'
WHERE user_correction IS NULL AND review_status = 'pending';

CREATE INDEX IF NOT EXISTS idx_feedback_memory_review_status
    ON cloud_feedback_memory (review_status, created_at DESC);

-- Only APPROVED feedback is returned to the agents.
-- (DROP first: the RETURNS TABLE shape changes, which CREATE OR REPLACE can't do.)
DROP FUNCTION IF EXISTS match_past_experience(vector, float, int);
CREATE FUNCTION match_past_experience (
  query_embedding vector(512),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id uuid,
  user_query text,
  system_response text,
  is_positive boolean,
  user_correction text,
  review_status varchar,
  similarity float
)
LANGUAGE plpgsql
SET search_path = public, extensions
AS $$
#variable_conflict use_variable
BEGIN
  RETURN QUERY
  SELECT
    cfm.id,
    cfm.user_query,
    cfm.system_response,
    cfm.is_positive,
    cfm.user_correction,
    cfm.review_status,
    1 - (cfm.query_embedding <=> query_embedding) AS similarity
  FROM cloud_feedback_memory cfm
  WHERE cfm.review_status = 'approved'
    AND 1 - (cfm.query_embedding <=> query_embedding) > match_threshold
  ORDER BY cfm.query_embedding <=> query_embedding ASC
  LIMIT match_count;
END;
$$;

ALTER FUNCTION match_cloud_documents(vector, float, int, text) SET search_path = public, extensions;

-- ---------------------------------------------------------------------------------------
-- 2. Row Level Security (no policies = deny for anon/authenticated)
-- ---------------------------------------------------------------------------------------
ALTER TABLE cloud_documents            ENABLE ROW LEVEL SECURITY;
ALTER TABLE cloud_competitor_mapping   ENABLE ROW LEVEL SECURITY;
ALTER TABLE cloud_service_pricing      ENABLE ROW LEVEL SECURITY;
ALTER TABLE cloud_feedback_memory      ENABLE ROW LEVEL SECURITY;
ALTER TABLE cloud_conversation_history ENABLE ROW LEVEL SECURITY;

-- ---------------------------------------------------------------------------------------
-- 3. RPC functions: backend only
-- ---------------------------------------------------------------------------------------
REVOKE EXECUTE ON FUNCTION match_past_experience(vector, float, int) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION match_cloud_documents(vector, float, int, text) FROM PUBLIC, anon, authenticated;
GRANT  EXECUTE ON FUNCTION match_past_experience(vector, float, int) TO service_role;
GRANT  EXECUTE ON FUNCTION match_cloud_documents(vector, float, int, text) TO service_role;

COMMIT;
