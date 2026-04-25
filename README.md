# Database Report Agent

A natural-language reporting agent over an enterprise-style Postgres database. Built with Chainlit (UI), LangGraph (orchestration), and Claude (reasoning), backed by Neon serverless Postgres loaded with the Pagila sample database. Personal portfolio project — the goal is end-to-end coverage of production agent patterns, not just a working demo.

## Goals

This project exists to demonstrate competence at every layer of building, evaluating, and deploying an agentic system. The agent itself (text-to-SQL + report generation) is the substrate — the interesting work is the surrounding production scaffolding: validation, retries, RAG, evals, feedback loops, caching, output validation, and tracing. Reviewers should be able to look at any layer and find a deliberate design choice, not a tutorial copy-paste.

The agent answers natural-language questions about a sales/rental database and returns SQL-backed answers with optional chart suggestions. It must never modify data — read-only is enforced at three levels (DB role, SQL validator, connection-level transaction mode).

## Architecture

Three layers, with a hard read-only boundary between the reasoning layer and the data layer:

```
Chainlit (UI, streaming, feedback)
    ↓
LangGraph agent (state machine, retry loop)
    ↓ Claude API for reasoning steps
SQL validator (sql_guard — code fence)
    ↓
Neon Postgres (Pagila + agent schema, read-only role for runtime)
```

### LangGraph node layout

```
classify_intent ──► handle_chat ──► END
        │
        ▼ (data)
generate_sql ──► validate_sql ──► execute_sql ──► summarize ──► END
        ▲              │                │
        │              │ error          │ error
        └──────────────┴────────────────┘
              (retries < MAX_RETRIES)
```

Six nodes, conditional edges driving retries. `generate_sql` and `execute_sql` both route back to `generate_sql` on error, with the specific error message in state — specific errors get fixed on retry, vague ones waste retries. Cap is 2 retries; on exhaustion, `summarize` degrades gracefully with the error in the response.

### State (AgentState TypedDict)

- `messages` — full conversation, uses LangGraph's `add_messages` reducer
- `question` — extracted current turn so downstream nodes don't re-parse
- `intent` — `"data"` | `"chat"` | `"unclear"`
- `sql` — generated query (overwritten by validator with canonicalized + LIMIT-injected version)
- `sql_error` — populated by validator or executor; signals retry path
- `rows` — query results as `list[dict]`
- `summary` — final natural-language answer
- `chart` — optional `ChartSpec` for Chainlit rendering
- `retries` — counter, lives in state (not closure) so it survives checkpointing

Presence/absence of fields signals pipeline position — useful for debugging traces.

## Tech stack and rationale

**Python 3.11+, uv for dependency management.** Standard modern setup.

**Chainlit** for the chat UI. Streams agent steps natively, has thumbs-up/down feedback primitives, runs as single-process so no infra surprises. Picked over HuggingFace Spaces (wrong fit for stateful DB-backed apps) and Chainlit Cloud (skipped for portfolio reasons — want to show real deployment).

**LangGraph** for orchestration. Picked structured graph over `create_sql_agent`'s ReAct pattern because: text-to-SQL has a deterministic pipeline (classify → generate → validate → execute → summarize), so letting the LLM re-derive it each turn costs 2-4 extra calls per question for nothing. Structured graphs also produce cleaner traces, which matters for the demo story. Used `MemorySaver` for v0; will swap to `PostgresSaver` for production.

**Claude (Sonnet 4.5)** as the reasoning model. `langchain-anthropic` provides the binding, with `with_structured_output()` for Pydantic-validated outputs from every LLM step.

**Neon serverless Postgres** for the database. Picked over Railway's bundled Postgres because (1) `pgvector` is a first-class supported extension on Neon (one `CREATE EXTENSION vector;`) and our RAG layer depends on it, (2) scale-to-zero means demo costs ≈$0, (3) branching is a real feature for eval sandboxes later. Same backup/connection story for the business data and the agent metadata — both in one DB, separated by schema.

**Pagila** as the sample dataset. Picked over generating fake data with Faker (statistically flat, boring queries), AdventureWorks (overkill, dated), and TPC-H (synthetic, schema feels academic). Pagila is the Postgres port of Sakila — DVD rental store, ~15 tables, realistic time-series and join patterns. Did not rename tables to feel less video-store-y; "imagine this is any catalog business" works fine for narrative, and renaming breaks every reference doc.

**SQLAlchemy + langchain-community's `SQLDatabase`** for DB access. `SQLDatabase.get_table_info()` gives prompt-ready DDL with sample rows and is annoying to rewrite. Don't use `SQLDatabase.run()` — its return format is stringified `repr()` output, useless for programmatic consumption. Use the engine directly via SQLAlchemy for execution, get typed dict rows back. SQLAlchemy is underneath `SQLDatabase` whether you import it directly or not.

**psycopg v3** as the Postgres driver. URLs normalized to `postgresql+psycopg://` at the SQLAlchemy boundary so `.env` stays in standard format.

**sqlglot** for SQL parsing in the validator. Pure Python, Postgres dialect support, AST walking. Don't try to validate SQL with regex.

**Railway** for hosting the Chainlit app. Picked over Render (cold starts on free tier kill demos), Fly.io (more setup), and HF Spaces (wrong shape). DB lives on Neon, app on Railway — clean separation, both providers have Git-deploy.

## Read-only enforcement (three fences)

This is the safety-critical part of the project.

**Fence 1: DB role.** `report_agent` role created via SQL with `NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT`. Granted only `CONNECT` on database, `USAGE` on `public` schema, `SELECT` on all tables, and `ALTER DEFAULT PRIVILEGES ... GRANT SELECT` for future tables. Revoked `ALL FROM PUBLIC` to plug that leak. **Critical Neon gotcha:** roles created via the Neon dashboard UI default to `neon_superuser` group membership and have full write access. Always create restricted roles via SQL, with explicit attribute list, then verify with `\du` and `pg_auth_members` query. Never trust the UI default.

**Fence 2: SQL validator (`sql_guard`).** Parses with sqlglot, asserts top-level is `Select` or `Union` (possibly wrapped in `With`), walks the AST rejecting any mutation node anywhere (catches DML inside CTEs — Postgres really supports `WITH deleted AS (DELETE ...) SELECT`), rejects single-statement only, rejects dangerous functions (`pg_read_file`, `dblink`, `lo_import`, etc.), injects `LIMIT 1000` if missing. Returns canonicalized SQL that the executor runs.

**Fence 3: Connection-level.** Engine `connect` event sets `statement_timeout = 10000` and `default_transaction_read_only = on` per session. **Neon pooler gotcha:** Neon's pooler (PgBouncer transaction mode) rejects startup-time `-c statement_timeout=...` parameters — must be set via `SET` after connection via the SQLAlchemy `connect` event hook, not via `connect_args.options`.

Any one fence failing alone doesn't compromise the read-only guarantee.

## Project structure

```
data-reporting-agent/
├── .env                      # gitignored
├── .env.example
├── pyproject.toml
├── README.md                 # this file
├── chainlit.md
├── app.py                    # Chainlit entrypoint
├── smoketest.py              # CLI end-to-end check
├── agent/
│   ├── __init__.py
│   ├── graph.py              # LangGraph assembly, app_graph singleton
│   ├── state.py              # AgentState, ChartSpec, initial_state()
│   ├── nodes.py              # all node functions
│   ├── prompts.py            # prompt templates (TODO)
│   ├── schemas.py            # Pydantic models for structured outputs (TODO)
│   ├── sql_guard.py          # validator
│   └── db.py                 # SQLAlchemy engines, SQLDatabase wrapper, run_query
└── tests/
    └── test_sql_guard.py
```

## Status

### Done

- Neon project provisioned, Pagila loaded into `neondb`
- `report_agent` role created with proper restrictions, verified via write-attempt tests
- `agent/db.py` complete: lazy SQLAlchemy engines (agent + admin), `SQLDatabase` wrapper for introspection, `run_query()` for typed dict results, `verify_connection()` for startup sanity check, `connect` event hook for session params
- `agent/state.py` complete: `AgentState` TypedDict, `ChartSpec`, `initial_state()`
- `agent/graph.py` complete: all six nodes wired, conditional edges, `MemorySaver` checkpointer, `app_graph` singleton exported
- `agent/nodes.py`: stubs for all six nodes; `execute_sql` is real (uses `run_query`)
- `app.py` minimal Chainlit entrypoint with session-keyed thread IDs
- `smoketest.py` runs end-to-end through stub graph, prints intent/SQL/rows/summary
- `agent/sql_guard.py` complete with `tests/test_sql_guard.py` test suite
- `validate_sql` node wired to use `sql_guard`

### Not done — in priority order

1. **`agent/schemas.py`** — Pydantic models for structured LLM outputs:
   - `IntentClassification` — `intent: Literal["data", "chat", "unclear"]`
   - `SQLGeneration` — `sql: str`, `tables_used: list[str]`, `reasoning: str`
   - `ReportOutput` — `summary: str`, `key_findings: list[str]`, `chart_suggestion: ChartSpec | None`
   Use `BaseModel` from pydantic v2.

2. **`agent/prompts.py`** — prompt templates as constants. System prompts for intent classification, SQL generation (includes schema string from `pagila_schema_string()`, with retry-aware "previous SQL had this error" branch), summarization, chat handling. Keep prompts as f-string templates or `langchain.prompts.PromptTemplate` — either works. Schema injected statically since Pagila's ~22 tables fit in context comfortably.

3. **Real LLM nodes** in `agent/nodes.py`. Replace stubs in this order:
   - `classify_intent` — easiest, uses `llm.with_structured_output(IntentClassification)`
   - `summarize` — formatting-heavy, uses `ReportOutput`. Receives rows + question, produces summary + chart spec
   - `handle_chat` — plain LLM call for non-data turns
   - `generate_sql` — hardest. Reads `prior_sql` and `prior_error` from state for retry context. Uses `SQLGeneration` schema.

   All four use `ChatAnthropic(model="claude-sonnet-4-5", temperature=0)`. Wrap each in try/except for Pydantic parse failures, route back via `sql_error` field for retry.

4. **Chainlit streaming.** Currently `app.py` uses `ainvoke` and shows only the final summary. Swap to `astream_events` and surface intermediate node steps to the user (`cl.Step` for each node). The streaming UX is half the demo's value — when a query takes 5 seconds, the user should see "classifying → generating SQL → validating → executing → summarizing" progress, not a frozen spinner.

5. **End-to-end smoke testing with real LLM.** Update `smoketest.py` to test 5-10 hand-picked questions across difficulty levels. Confirm the retry loop actually triggers and recovers (induce by giving the LLM a question about a non-existent column, watch retry context kick in).

### Layered additions, in build order

After v0 happy path works end-to-end with real LLMs:

6. **Evals.** This goes before RAG — without measurement, can't tell if RAG helps. Eval set of 30-50 cases as JSON or in `agent.eval_cases` table. Two graders stacked: execution-equivalence on SQL (run both queries, hash-compare result sets — much better than string match), LLM-as-judge for summary quality with three-part rubric (factual accuracy, relevance, format). Harness is a Python script that loops cases, writes to `agent.eval_runs`. Headline number matters less than the failure-mode breakdown table — that's what tells you what to fix.

7. **`agent` schema setup.** Create `agent` schema in Neon, enable `pgvector`, create tables: `few_shot_examples`, `glossary`, `eval_cases`, `eval_runs`, `feedback_candidates`, `result_cache`. HNSW indexes on embedding columns once corpora exceed ~500 rows. Update grants: `report_agent` gets SELECT on `agent.*` plus INSERT/UPDATE/DELETE only on `result_cache` and INSERT on `feedback_candidates`.

8. **RAG.** Two pipelines:
   - **Few-shot example retrieval.** Hand-seed 30-50 (question, SQL) pairs spanning aggregates, time-series, self-joins, top-N. Embed *questions*, not SQL. Top-3 vector search, optional reranking via Cohere Rerank or Claude Haiku. New node `retrieve_examples` between `classify_intent` and `generate_sql` on the data path.
   - **Glossary retrieval.** Hand-seed 10-20 entries encoding semantic landmines ("active customer means rental in last 90 days," "revenue is `sum(quantity * unit_price * (1 - discount))` from `order_items`"). Hybrid search: BM25 (Postgres `tsvector`) + vector, combined via Reciprocal Rank Fusion, gated by similarity threshold so irrelevant terms don't pollute the prompt. New node `retrieve_glossary` parallel to `retrieve_examples`.
   - Embeddings via Voyage-3-lite or OpenAI `text-embedding-3-small`, both 1536-dim. pgvector cosine distance.
   - **HyDE optional**: have LLM generate hypothetical SQL, embed alongside question, search with both. Worth it once base RAG is working and evals show retrieval is the bottleneck.
   - Prompt assembly order: system prompt → schema DDL → relevant glossary → few-shot examples → user question. Instructions first, question last.

9. **Feedback loop.** Chainlit thumbs-up/down callback writes `(question, sql, result_hash, rating, optional_correction)` to `agent.feedback_candidates`. Manual review for v1 — promote good ones to `few_shot_examples`. Critical subtlety: thumbs-up means user was satisfied, not that SQL was correct. Re-execute and sanity-check before promotion.

10. **Caching.** Three layers in decreasing ROI:
    - Schema cache (already done implicitly via `@cache` on `pagila_schema_string`)
    - Result cache: `(normalized_sql, date_bucket)` → rows, 15min TTL. Canonicalize SQL with sqlglot first. Hits write to `agent.result_cache`. Inside `execute_sql`, check before hitting DB.
    - Skip semantic question cache for v1 — staleness risk too high without careful gating.

11. **Output validation.** Already partially done via `with_structured_output`. Add layered structural checks after Pydantic parsing: `sqlglot.parse_one()` succeeds, statement is SELECT-only (already in guard), referenced tables exist in schema, LIMIT present (injected by guard). Specific error → retry; vague error wastes retries.

12. **Tracing.** LangSmith via `LANGCHAIN_TRACING_V2=true` + `LANGCHAIN_API_KEY` in `.env`. Free tier is plenty. Langfuse is the OSS alternative if vendor lock-in matters. The artifact for the demo is a trace screenshot of a complex question hitting the retry loop and recovering — much better than prose about observability.

13. **Deployment.** Railway for Chainlit app, Neon already hosts DB. Custom domain optional, Cloudflare in front for caching + DDoS optional. `.env` keys go in Railway Variables tab. Set spending cap on Railway since billing is usage-based.

## Important environment

`.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL_AGENT=postgresql://report_agent:...@ep-xyz-pooler.region.aws.neon.tech/neondb?sslmode=require
DATABASE_URL_ADMIN=postgresql://neondb_owner:...@ep-xyz.region.aws.neon.tech/neondb?sslmode=require
VOYAGE_API_KEY=pa-...        # for embeddings, when RAG layer goes in
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_PROJECT=db-agent-dev
```

Note: agent URL uses pooler endpoint (`-pooler` suffix), admin URL does not. Pooler doesn't support session-level features like `SET ROLE` across transactions, so admin/migration work needs the direct endpoint.

## Things to remember when continuing

- Database is `neondb`, not `pagila` — Neon's default DB name. Pagila is just loaded into it.
- 22 tables visible (15 base + 7 views). `SQLDatabase` includes views by default, that's fine.
- `report_agent` connection is pooler endpoint with timeout/readonly via `SET`, not URL options.
- Pagila data is from 2005-2006 — eval cases need to specify years explicitly or the demo needs an "as of 2006" framing.
- `MAX_RETRIES = 2` in `agent/nodes.py`. Failures route back to `generate_sql` with error in state; on exhaustion, route forward to `summarize` for graceful degradation.
- The graph is checkpointed via `MemorySaver` — process restart loses conversation state. Swap to `PostgresSaver` once it matters.
- Don't add tool-calling/ReAct loops. The flow is deterministic; structured edges are easier to reason about, debug, and eval. ReAct only earns its keep when there are non-trivial tools whose call order isn't fixed.
- Don't proactively rename Pagila tables to feel more enterprise — breaks every reference doc, narrative framing handles it.
- Do not use `psycopg2` — the project uses psycopg v3, normalized at the SQLAlchemy URL boundary.
- Do not use `SQLDatabase.run()` for query execution — its return format is stringified. Use `run_query()` from `db.py`.
- Verify role permissions whenever touching grants. The Neon dashboard UI creates roles with too many privileges; always re-check with `\du` and the `pg_auth_members` membership query.

## References

- Pagila: https://github.com/devrimgunduz/pagila
- LangGraph SQL agent tutorial (study, don't copy): https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
- Neon docs: https://neon.tech/docs
- Chainlit docs: https://docs.chainlit.io
