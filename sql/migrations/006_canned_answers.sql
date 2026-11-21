-- Migration 006 — reviewed canned answers (run after 005; idempotent).
-- Frequently asked questions with a human-approved answer. A matching question is answered
-- from here without calling an LLM; users see an ordinary answer. Only status='approved'
-- rows that have not expired are ever returned. Backend (service_role) only.

BEGIN;

CREATE TABLE IF NOT EXISTS public.cloud_canned_answers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    route VARCHAR(20) NOT NULL DEFAULT 'document'
        CHECK (route IN ('calculator', 'document', 'general')),
    question_embedding VECTOR(512) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'approved', 'disabled')),
    source TEXT,                      -- where the answer's facts come from
    reviewed_by TEXT,
    approved_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,   -- NULL = no expiry
    hit_count INTEGER NOT NULL DEFAULT 0,
    last_hit_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT timezone('utc'::text, now())
);

CREATE INDEX IF NOT EXISTS idx_canned_answers_embedding
    ON public.cloud_canned_answers USING hnsw (question_embedding vector_cosine_ops);

ALTER TABLE public.cloud_canned_answers ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION public.match_canned_answer (
    query_embedding vector(512),
    match_threshold float,
    match_count int DEFAULT 3
)
RETURNS TABLE (id uuid, question text, answer text, route varchar, similarity float)
LANGUAGE sql STABLE
SET search_path = public, extensions
AS $$
    SELECT ca.id, ca.question, ca.answer, ca.route,
           1 - (ca.question_embedding <=> query_embedding) AS similarity
    FROM public.cloud_canned_answers ca
    WHERE ca.status = 'approved'
      AND (ca.expires_at IS NULL OR ca.expires_at > now())
      AND 1 - (ca.question_embedding <=> query_embedding) >= match_threshold
    ORDER BY ca.question_embedding <=> query_embedding
    LIMIT greatest(1, least(match_count, 10));
$$;

CREATE OR REPLACE FUNCTION public.record_canned_hit (answer_id uuid)
RETURNS void
LANGUAGE sql
SET search_path = public
AS $$
    UPDATE public.cloud_canned_answers
    SET hit_count = hit_count + 1, last_hit_at = now()
    WHERE id = answer_id;
$$;

REVOKE EXECUTE ON FUNCTION public.match_canned_answer(vector, float, int) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.record_canned_hit(uuid) FROM PUBLIC, anon, authenticated;
GRANT  EXECUTE ON FUNCTION public.match_canned_answer(vector, float, int) TO service_role;
GRANT  EXECUTE ON FUNCTION public.record_canned_hit(uuid) TO service_role;

COMMIT;
