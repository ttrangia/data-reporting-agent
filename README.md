# Database reporting agent

A natural-language reporting agent over a Postgres database, built with **LangGraph + Claude + Chainlit** and deployed on Railway. Ask natural-language questions, get SQL-backed answers and charts. Multi-section reports fan out into parallel sub-queries.

**Live demo:** [pagila.tinotrangia.com](https://pagila.tinotrangia.com) — DM for credentials.

<https://github.com/user-attachments/assets/355aad09-7d7c-4c9f-869f-6db8b9ce7702>

*Single-question flow — pipeline steps light up live, prose streams in, chart renders alongside.*

<https://github.com/user-attachments/assets/160aebd5-c930-49bd-b0e2-e71f199b142d>

*Multi-section report — parallel sub-queries fan out concurrently, compose into a single streamed markdown.*



*Multi-section report — parallel sub-queries fan out concurrently, compose into a single streamed markdown.*

The substrate is a Pagila DVD-rental sample database; the interesting work is the production scaffolding around it: routing, retries, RAG, sandboxed code exec, evals, prompt-cache optimization, and graceful degradation everywhere.

---

## What it does

- **Answers single questions** with one SQL query, streamed prose, and a Plotly chart when one fits.
- **Decomposes multi-part asks** ("top films *and* revenue trend") into parallel sub-queries that compose into a tight report.
- **Generates open-ended reports** from broad asks ("quarterly review") — fans out 4-7 sections covering headline metrics, trends, top entities, breakdowns.
- **Re-charts** prior results without re-querying ("show that as a pie chart") — multi-turn memory via LangGraph's checkpointer.
- **Refuses** prompt-injection, bulk-PII, and over-length inputs at a deterministic safety gate before any LLM call.
- **Diagnoses empty results** — when SQL returns 0 rows, runs a follow-up query that surfaces what values *do* have data, instead of a flat "no rows".

## Architecture

```
                  ┌─────────────────┐
                  │   front_agent   │  classify intent + safety gate
                  └────────┬────────┘
              data ┌───────┴────────┐ report
                   ▼                ▼
        ┌──────────────────┐  ┌────────────────┐
        │ retrieve_context │  │  plan_report   │  decompose into N sections
        │  (RAG: glossary  │  └────────┬───────┘
        │  + few-shot SQL) │           ▼
        └────────┬─────────┘  ┌────────────────┐
                 ▼            │ warmup_sql_    │  pre-warm prompt cache
        ┌──────────────────┐  │ cache          │  before parallel fan-out
        │   generate_sql   │  └────────┬───────┘
        │ ◄── retry on err │           ▼
        └────────┬─────────┘     ┌──────────────┐
                 ▼               │  sub_query   │  parallel × N (Send API)
        ┌──────────────────┐     │ (RAG inline) │
        │   validate_sql   │     └──────┬───────┘
        │  (sqlglot guard) │            ▼
        └────────┬─────────┘     ┌──────────────────┐
                 ▼               │ aggregate_report │  streams composed
        ┌──────────────────┐     └──────┬───────────┘  markdown
        │   execute_sql    │            ▼
        └────────┬─────────┘           END
                 ▼
   (0 rows) ┌──────────────┐
        ┌───┤ diagnose_    │
        │   │ empty        │
        │   └──────┬───────┘
        ▼          ▼
        ┌──────────────────┐  streams prose tokens
        │    summarize     │  to user-facing message
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐  LLM-authored Plotly code,
        │  generate_chart  │  AST-validated sandbox
        └────────┬─────────┘
                 ▼
                END
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph 1.x** | Structured state machine; parallel fan-out via Send API; first-class checkpointing |
| Reasoning | **Claude Sonnet 4.6** + **Haiku 4.5** | Sonnet for SQL/planning/charts; Haiku for streaming prose; prompt caching cuts repeat cost ~10× |
| RAG | **Voyage `voyage-3`** + **pgvector** | Hand-curated glossary + few-shot SQL corpus; cosine similarity in SQL |
| UI | **Chainlit 2.x** | Native streaming, per-step indicators, Plotly element support |
| Database | **Neon Postgres** | Serverless, scale-to-zero, pgvector built in, branchable |
| Tracing | **LangSmith** | Per-node traces, eval-suite tagging, free tier is generous |
| Deploy | **Railway** + Docker | Git-deploy, custom domain, autoTLS |

## Quick start

```bash
git clone <this-repo> && cd data-reporting-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in keys (see file for what's needed)

# One-time RAG setup
psql "$DATABASE_URL_ADMIN" -f agent/rag/schema.sql
python -m agent.rag.build_index

# Run
chainlit run app.py
```

Open `http://localhost:8000`, log in with `APP_USERNAME` / `APP_PASSWORD`.

## Project layout

```
agent/
  graph.py            LangGraph wiring — nodes, edges, conditional routing
  nodes.py            13 graph nodes (front_agent, generate_sql, sub_query, ...)
  state.py            AgentState TypedDict, custom reducers, ChartCode
  schemas.py          Pydantic models for structured LLM output
  prompts.py          All prompts (cached system prefixes, dataset notes)
  db.py               Postgres engine, schema string, vocabulary enumeration
  sql_guard.py        sqlglot-based read-only validator
  chart_sandbox.py    AST-validated Plotly code execution
  safety.py           Deterministic input refusal patterns
  rag/
    glossary.yaml     Business terms, conventions, dataset quirks
    examples.yaml     Hand-curated (question, sql, notes) triplets
    build_index.py    Offline indexer — embeds, hashes, upserts to pgvector
    retrieve.py       Runtime retrieval — embed + cosine top-K

evals/
  dataset.yaml        12 starter cases across data/respond/rechart/report/safety
  predicates.py       21 pluggable checks (deterministic + LLM-judge)
  runner.py           CLI runner, parallel cases, markdown scorecard
  scorecard.py        Per-case + per-predicate failure breakdown
  reports/            Timestamped scorecards committed for quality drift

tests/                160 unit tests — graph topology, retry loops, sandbox isolation
app.py                Chainlit entry point — streaming, auth, step lifecycle
chainlit.md           Welcome-screen content (renders in the readme tab)
public/icons/         Starter-bubble icons + assistant avatar
Dockerfile            Python 3.12-slim, layered for cache hits
requirements.txt      Pinned (178 packages)
```

## Read-only safety, three fences

1. **DB role.** `report_agent` Postgres role created via SQL with explicit attribute revokes. Granted `CONNECT`, `USAGE` on `public`, `SELECT` on tables — nothing else. Statement timeout set per-connection. (Critical Neon gotcha: roles created via the dashboard UI default to `neon_superuser`. Always create restricted roles via SQL.)
2. **SQL guard.** `agent/sql_guard.py` parses every generated query with sqlglot, walks the AST, rejects anything that isn't a top-level SELECT or UNION. Mutation nodes (INSERT, UPDATE, DELETE, DDL, COPY) are denylisted *anywhere* in the tree (catches DML inside CTEs). Canonicalizes the SQL and injects a default `LIMIT` before returning to the executor.
3. **Connection-level.** Every connection runs `SET default_transaction_read_only = on` at session start. (Neon pooler gotcha: must be set via `SET` after connect, not as a startup parameter — pooler in transaction mode rejects those.)

Plus deterministic input refusal in `agent/safety.py` (prompt injection, bulk PII, over-length) ahead of the LLM. Any one fence failing alone doesn't compromise the read-only guarantee.

## Key design decisions

- **Structured graph over ReAct.** Text-to-SQL has a deterministic pipeline (classify → retrieve → generate → validate → execute → summarize). ReAct re-derives this every turn for 2-4 extra LLM calls' worth of nothing. The graph also produces cleaner traces.
- **LLM-authored Plotly code in a sandbox**, not a fixed `ChartSpec` schema. Lets the model use the full `px` / `go` / `make_subplots` surface — at chart quality on par with what Claude.app produces in its sandbox. Safety: AST-level validation (no imports, no dunder access, denylisted names), curated builtin allowlist, thread-pool timeout. Demo-grade isolation, not adversarial.
- **Cache pre-warming for parallel fan-out.** Before report sub-queries fire in parallel, a single warmup call writes the SQL prompt cache. Without this, all N parallel branches see no cache yet and each pay the 1.25× write premium on the same 8.9k-token prefix. With it: 1 write + N-1 reads. Cuts report token spend ~25%.
- **Variable retrieved-context in the user message, not the system prompt.** Keeps the system prefix stable across requests so cache hits work. Question-specific RAG block goes in the user message — costs 500-1500 tokens per call but at non-cache-write rates.
- **Three-layer LLM transport hardening.** SDK request timeout (30s) + SDK retries (2 attempts) + outer `asyncio.wait_for` ceiling (90s). Defends against three different stall modes that have all bitten in production.
- **Eval scorecards committed to git.** Every run writes `evals/reports/<date>_<sha>.md`. `git log --follow evals/reports/` shows quality drift over time — a much better quality artifact than a Notion page.

## Eval harness

```bash
python -m evals.runner                  # full suite, real DB + LLM (~$1)
python -m evals.runner --id top_films   # single case
python -m evals.runner --tag baseline   # label run for LangSmith filtering
python -m evals.runner --concurrency 1  # serial, easier to debug
```

Each case is a YAML entry with predicates against the final agent state. 21 predicate types: structural (`intent_equals`, `sql_references_all`), result-shape (`rows_count_at_least`), prose (`summary_mentions_any`, `summary_satisfies` — LLM-judge), chart (`chart_code_contains`), report (`report_section_count_at_least`, `report_all_sections_succeeded`).

Multi-turn cases supported (rechart flow). Each case becomes a clickable trace in LangSmith via the `eval` + `case:<id>` tags.

## Tracing

LangSmith tracing is on by default when `LANGSMITH_TRACING=true` is set. Each turn is tagged by source (`chainlit` / `eval`) and `run_name` carries the question text — the dashboard becomes filterable by both. Per-node traces show prompts, completions, and token counts including cache hits.

## Deployment

[Live demo here](https://pagila.tinotrangia.com).

- **App**: Railway, Docker build (`Dockerfile` in repo). Auto-deploys on push to `main`.
- **Database**: Neon, scale-to-zero. The agent's connection-init hook handles cold-start retries.
- **Auth**: Single shared password gate via Chainlit's `password_auth_callback` — enough to keep crawlers off and prevent random API-cost burn.
- **Custom domain**: `pagila.tinotrangia.com` via CNAME, autoTLS via Let's Encrypt.

## Things not built (deliberate scope cuts)

- **No `psycopg2`** — psycopg v3, normalized at the SQLAlchemy URL boundary.
- **No `SQLDatabase.run()`** — its return type is stringified. `run_query()` in `db.py` returns typed `dict` rows.
- **No ReAct/tool-calling loop** — see "design decisions" above.
- **No multi-user system** — single shared password. Upgrade to OAuth before serving real users.
- **No CI** — pre-push runs the test suite manually.
- **No `PostgresSaver`** — `MemorySaver` is sufficient for portfolio scope. Conversation state is lost on restart.

## Pagila note

This Pagila install has all rental/payment activity dated **February–August 2022** (not 2005-2006 as the upstream README says). All operational store data is concentrated at exactly **2 stores via the inventory join path**: store 1 (Boksburg, South Africa) and store 2 (Hamilton, New Zealand). The 500-store catalog is metadata only. The agent's `DATASET_NOTES` block + glossary entry encode this so it doesn't hallucinate joins through `customer.store_id` (a legacy FK) or `staff.store_id` (different stores, same money attributed differently).

## License

MIT.
