# agent/db.py
import logging
import os
import time
from functools import cache
from typing import Callable, TypeVar

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Neon scale-to-zero cold starts can drop the first connection mid-query.
# Retry transient OperationalError with brief backoff before surfacing.
COLD_START_RETRIES = 3
COLD_START_BACKOFF_S = (0.5, 1.5, 3.0)


def _with_cold_start_retry(fn: Callable[[], T]) -> T:
    last_exc: OperationalError | None = None
    for attempt in range(COLD_START_RETRIES):
        try:
            return fn()
        except OperationalError as e:
            last_exc = e
            if attempt + 1 == COLD_START_RETRIES:
                break
            sleep_s = COLD_START_BACKOFF_S[attempt]
            logger.warning(
                "DB OperationalError on attempt %d, retrying in %.1fs: %s",
                attempt + 1, sleep_s, e,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _make_engine(url: str, *, readonly: bool) -> Engine:
    # Normalize to psycopg v3 driver
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)

    engine = create_engine(
        url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=1800,
    )

    # Set session parameters per-connection, after startup — pooler-safe
    @event.listens_for(engine, "connect")
    def _set_session_params(dbapi_conn, _):
        with dbapi_conn.cursor() as cur:
            cur.execute("SET statement_timeout = 10000")
            if readonly:
                cur.execute("SET default_transaction_read_only = on")

    return engine


@cache
def agent_engine() -> Engine:
    return _make_engine(os.environ["DATABASE_URL_AGENT"], readonly=True)


@cache
def admin_engine() -> Engine:
    return _make_engine(os.environ["DATABASE_URL_ADMIN"], readonly=False)


@cache
def agent_db() -> SQLDatabase:
    return SQLDatabase(engine=agent_engine(), sample_rows_in_table_info=3)


@cache
def pagila_schema_string() -> str:
    """DDL + sample rows for every Pagila table, ready to drop into a prompt."""
    return _with_cold_start_retry(lambda: agent_db().get_table_info())


# (Table, column) pairs to pre-enumerate at boot. Picked for low cardinality
# (< ~200 distinct values) and high "WHERE clause" usefulness — exactly the
# values a SQL generator is most likely to misspell or hallucinate ("United
# States" vs the actual canonical spelling, "Action" vs "action" category,
# etc.). The 3-sample-row schema dump can't surface these.
_VOCABULARY_COLUMNS: list[tuple[str, str]] = [
    ("country",  "country"),
    ("category", "name"),
    ("language", "name"),
    ("film",     "rating"),
    ("film",     "rental_rate"),
]


@cache
def low_cardinality_vocab() -> dict[str, list[str]]:
    """Distinct values for each (table, column) in _VOCABULARY_COLUMNS.

    Loaded once at boot, cached for the process lifetime. Lets the SQL
    generator use exact spellings in WHERE clauses instead of guessing.
    Returns a dict keyed by 'table.column' → sorted list of stringified values.
    Skips a column silently on error rather than failing the whole load."""
    def _load() -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        with agent_engine().connect() as conn:
            for table, col in _VOCABULARY_COLUMNS:
                try:
                    rows = conn.execute(text(
                        f'SELECT DISTINCT "{col}" AS v FROM "{table}" '
                        f'WHERE "{col}" IS NOT NULL ORDER BY "{col}"'
                    )).all()
                    result[f"{table}.{col}"] = [str(r.v) for r in rows]
                except Exception as e:
                    logger.warning("Skipping vocab for %s.%s: %s", table, col, e)
        return result
    return _with_cold_start_retry(_load)


@cache
def vocabulary_string() -> str:
    """Format low_cardinality_vocab() as a markdown block for prompt injection.

    Lists ALL values for each enumerated column — even ~109 countries fits
    comfortably in the prompt-cached static prefix. Better to list everything
    than to truncate and risk the value the user wants being hidden."""
    vocab = low_cardinality_vocab()
    if not vocab:
        return ""
    lines = [
        "Known categorical values in the data — use these EXACT spellings in WHERE clauses, do not guess:",
        "",
    ]
    for col, values in vocab.items():
        n = len(values)
        joined = ", ".join(f'"{v}"' for v in values)
        lines.append(f"- `{col}` ({n} distinct value{'s' if n != 1 else ''}): {joined}")
    return "\n".join(lines)


def run_query(sql: str) -> list[dict]:
    def _run():
        with agent_engine().connect() as conn:
            result = conn.execute(text(sql))
            return [dict(row._mapping) for row in result]
    return _with_cold_start_retry(_run)


def verify_connection() -> dict:
    def _check():
        with agent_engine().connect() as conn:
            row = conn.execute(text(
                "SELECT current_user, current_database(), "
                "current_setting('statement_timeout') AS timeout_ms, "
                "current_setting('default_transaction_read_only') AS readonly"
            )).one()
            return dict(row._mapping)
    return _with_cold_start_retry(_check)


def warmup() -> None:
    """Wake Neon and prime caches so the first user query is fast."""
    verify_connection()
    pagila_schema_string()
    vocabulary_string()  # also prime the categorical-vocab cache