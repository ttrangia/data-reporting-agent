-- pgvector-backed RAG corpus for the data-reporting agent.
--
-- One row per curated YAML entry (glossary term or example query).
-- The indexer at agent/rag/build_index.py reads YAML, hashes the
-- embedding-text projection, and upserts when the hash changes.
-- Retrieval is one ORDER BY embedding <=> $1 LIMIT k per kind.
--
-- Run once on a fresh database (with the admin role, NOT report_agent):
--   psql "$DATABASE_URL_ADMIN" -f agent/rag/schema.sql
-- Or pipe via your tool of choice. Idempotent — safe to re-run.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_embeddings (
    id              TEXT PRIMARY KEY,           -- e.g. 'glossary:active_customer'
    kind            TEXT NOT NULL,              -- 'glossary' | 'example'
    content_hash    TEXT NOT NULL,              -- sha256 of embed_text — incremental rebuilds
    embedding       vector(1024) NOT NULL,      -- voyage-3 dim
    payload         JSONB NOT NULL,             -- full YAML entry (term/definition OR question/sql/notes)
    embedded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS rag_embeddings_kind_idx ON rag_embeddings (kind);

-- Grant read access to the agent role so retrieval works at query time.
-- The admin role owns + writes; report_agent only reads.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'report_agent') THEN
        GRANT SELECT ON rag_embeddings TO report_agent;
    END IF;
END$$;

-- HNSW index — uncomment when corpus exceeds ~500 entries.
-- For <500 rows, a sequential scan is faster than the index walk.
-- CREATE INDEX IF NOT EXISTS rag_embeddings_embedding_idx
--     ON rag_embeddings USING hnsw (embedding vector_cosine_ops);
