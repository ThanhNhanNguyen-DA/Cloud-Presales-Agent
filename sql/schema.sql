-- Enable pgvector extension to store and query vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Table 1: cloud_documents — stores technical docs, SLA, security, and compliance content for RAG
CREATE TABLE cloud_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(100),
    category VARCHAR(50),
    content TEXT NOT NULL,
    embedding VECTOR(512),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX idx_cloud_documents_embedding ON cloud_documents
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX idx_cloud_documents_category ON cloud_documents (category);

-- Table 2: cloud_competitor_mapping — competitor service mapping matrix
CREATE TABLE cloud_competitor_mapping (
    id SERIAL PRIMARY KEY,
    competitor_name VARCHAR(50) NOT NULL,
    competitor_service_name VARCHAR(100) NOT NULL,
    local_service_name VARCHAR(100) NOT NULL,
    feature_comparison TEXT,
    migration_difficulty VARCHAR(20) CHECK (migration_difficulty IN ('Easy', 'Medium', 'Hard')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX idx_competitor_mapping_competitor ON cloud_competitor_mapping (competitor_name);

-- Table 3: cloud_service_pricing — VM/Storage/Network pricing table
CREATE TABLE cloud_service_pricing (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100) NOT NULL,
    flavor_name VARCHAR(50) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    price_per_unit NUMERIC(12, 4) NOT NULL,
    currency VARCHAR(10) DEFAULT 'VND',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Table 4: cloud_feedback_memory — Episodic Feedback Memory, stores Upvote/Downvote events and human corrections
CREATE TABLE cloud_feedback_memory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_query TEXT NOT NULL,
    system_response TEXT NOT NULL,
    is_positive BOOLEAN DEFAULT TRUE,
    user_correction TEXT,
    query_embedding VECTOR(512),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

CREATE INDEX idx_feedback_memory_embedding ON cloud_feedback_memory
    USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);

-- RPC Function: match_past_experience — computes Cosine Similarity to find similar past lessons learned
CREATE OR REPLACE FUNCTION match_past_experience (
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
  similarity float
)
LANGUAGE plpgsql
AS $$
-- The `query_embedding` parameter shares its name with cloud_feedback_memory.query_embedding;
-- this tells PL/pgSQL to resolve the bare name to the parameter (the column is always
-- referenced qualified as cfm.query_embedding).
#variable_conflict use_variable
BEGIN
  RETURN QUERY
  SELECT
    cfm.id,
    cfm.user_query,
    cfm.system_response,
    cfm.is_positive,
    cfm.user_correction,
    1 - (cfm.query_embedding <=> query_embedding) AS similarity
  FROM cloud_feedback_memory cfm
  WHERE 1 - (cfm.query_embedding <=> query_embedding) > match_threshold
  ORDER BY cfm.query_embedding <=> query_embedding ASC
  LIMIT match_count;
END;
$$;

-- RPC Function: match_cloud_documents — same idea but for RAG document search
CREATE OR REPLACE FUNCTION match_cloud_documents (
  query_embedding vector(512),
  match_threshold float,
  match_count int,
  filter_category text DEFAULT NULL
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
  ORDER BY cd.embedding <=> query_embedding ASC
  LIMIT match_count;
END;
$$;