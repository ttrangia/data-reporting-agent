"""Pluggable predicates for the eval harness.

Each predicate is a pure function `(arg, agent_state) -> PredicateResult`.
Register one with `@predicate("name")`. The runner looks up predicates by
name from the YAML dataset's `expected:` blocks, so adding a new check is
two steps:
    1. `@predicate("my_check") def _my_check(arg, state): ...`
    2. Reference it in dataset.yaml under any case's `expected:` map.

Predicate args come from YAML — strings, ints, bools, or lists. The
predicate function is responsible for interpreting its own arg shape.

Failure messages should be short and self-contained — they land directly
in the scorecard table so a reader can diagnose without re-running.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any, Callable

import sqlglot
from sqlglot import expressions as exp
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class PredicateResult:
    name: str
    passed: bool
    reason: str  # short failure reason; empty when passed


PredicateFn = Callable[[Any, dict], PredicateResult]
_REGISTRY: dict[str, PredicateFn] = {}


def predicate(name: str):
    def deco(fn: PredicateFn) -> PredicateFn:
        if name in _REGISTRY:
            raise ValueError(f"Predicate {name!r} already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> PredicateFn:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown predicate: {name!r}. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


# ─── helpers ──────────────────────────────────────────────────────────────

def _ok(name: str) -> PredicateResult:
    return PredicateResult(name, True, "")


def _fail(name: str, reason: str) -> PredicateResult:
    return PredicateResult(name, False, reason)


def _extract_tables(sql: str | None) -> set[str]:
    """Best-effort table extraction. Returns lowercase, unqualified names.
    Falls back to empty set on parse error — predicate downgrades to
    'tables not found' rather than crashing the case."""
    if not sql:
        return set()
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
    except Exception:
        return set()
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name}


def _chart_code(state: dict) -> str | None:
    chart = state.get("chart")
    if chart is None:
        return None
    code = getattr(chart, "code", None)
    if code is None and isinstance(chart, dict):
        code = chart.get("code")
    return code


def _summary_lower(state: dict) -> str:
    return (state.get("summary") or "").lower()


# ─── routing & structure ─────────────────────────────────────────────────

@predicate("intent_equals")
def _intent_equals(expected: str, state: dict) -> PredicateResult:
    actual = state.get("intent")
    if actual == expected:
        return _ok("intent_equals")
    return _fail("intent_equals", f"expected {expected!r}, got {actual!r}")


@predicate("intent_in")
def _intent_in(expected: list[str], state: dict) -> PredicateResult:
    actual = state.get("intent")
    if actual in expected:
        return _ok("intent_in")
    return _fail("intent_in", f"expected one of {expected}, got {actual!r}")


@predicate("retries_at_most")
def _retries_at_most(n: int, state: dict) -> PredicateResult:
    actual = state.get("retries", 0) or 0
    if actual <= n:
        return _ok("retries_at_most")
    return _fail("retries_at_most", f"expected ≤{n}, got {actual}")


# ─── SQL ─────────────────────────────────────────────────────────────────

@predicate("sql_executed_successfully")
def _sql_ok(_: Any, state: dict) -> PredicateResult:
    if state.get("sql_error"):
        return _fail("sql_executed_successfully", f"sql_error: {state['sql_error']}")
    if state.get("rows") is None:
        return _fail("sql_executed_successfully", "rows is None (no SQL ran or SQL failed)")
    return _ok("sql_executed_successfully")


@predicate("sql_references_all")
def _sql_refs_all(tables: list[str], state: dict) -> PredicateResult:
    found = _extract_tables(state.get("sql"))
    needed = {t.lower() for t in tables}
    missing = needed - found
    if missing:
        return _fail(
            "sql_references_all",
            f"missing tables: {sorted(missing)} (saw: {sorted(found)})",
        )
    return _ok("sql_references_all")


@predicate("sql_references_none")
def _sql_refs_none(tables: list[str], state: dict) -> PredicateResult:
    found = _extract_tables(state.get("sql"))
    forbidden = {t.lower() for t in tables}
    leaked = found & forbidden
    if leaked:
        return _fail("sql_references_none", f"unexpectedly references: {sorted(leaked)}")
    return _ok("sql_references_none")


@predicate("sql_contains")
def _sql_contains(needle: str, state: dict) -> PredicateResult:
    sql = (state.get("sql") or "").lower()
    if needle.lower() in sql:
        return _ok("sql_contains")
    return _fail("sql_contains", f"sql does not contain {needle!r}")


# ─── rows ────────────────────────────────────────────────────────────────

@predicate("rows_count_at_least")
def _rows_at_least(n: int, state: dict) -> PredicateResult:
    rows = state.get("rows") or []
    if len(rows) >= n:
        return _ok("rows_count_at_least")
    return _fail("rows_count_at_least", f"expected ≥{n}, got {len(rows)}")


@predicate("rows_count_at_most")
def _rows_at_most(n: int, state: dict) -> PredicateResult:
    rows = state.get("rows") or []
    if len(rows) <= n:
        return _ok("rows_count_at_most")
    return _fail("rows_count_at_most", f"expected ≤{n}, got {len(rows)}")


@predicate("rows_count_equals")
def _rows_equals(n: int, state: dict) -> PredicateResult:
    rows = state.get("rows") or []
    if len(rows) == n:
        return _ok("rows_count_equals")
    return _fail("rows_count_equals", f"expected ={n}, got {len(rows)}")


@predicate("rows_nonempty")
def _rows_nonempty(_: Any, state: dict) -> PredicateResult:
    rows = state.get("rows") or []
    if rows:
        return _ok("rows_nonempty")
    return _fail("rows_nonempty", "rows is empty")


@predicate("rows_empty")
def _rows_empty(_: Any, state: dict) -> PredicateResult:
    rows = state.get("rows") or []
    if not rows:
        return _ok("rows_empty")
    return _fail("rows_empty", f"expected 0 rows, got {len(rows)}")


# ─── summary text ────────────────────────────────────────────────────────

@predicate("summary_present")
def _summary_present(_: Any, state: dict) -> PredicateResult:
    if (state.get("summary") or "").strip():
        return _ok("summary_present")
    return _fail("summary_present", "summary is empty")


@predicate("summary_mentions_any")
def _summary_any(needles: list[str], state: dict) -> PredicateResult:
    txt = _summary_lower(state)
    if any(n.lower() in txt for n in needles):
        return _ok("summary_mentions_any")
    return _fail("summary_mentions_any", f"none of {needles} in summary")


@predicate("summary_mentions_all")
def _summary_all(needles: list[str], state: dict) -> PredicateResult:
    txt = _summary_lower(state)
    missing = [n for n in needles if n.lower() not in txt]
    if not missing:
        return _ok("summary_mentions_all")
    return _fail("summary_mentions_all", f"missing from summary: {missing}")


@predicate("summary_mentions_none")
def _summary_none(needles: list[str], state: dict) -> PredicateResult:
    txt = _summary_lower(state)
    leaked = [n for n in needles if n.lower() in txt]
    if not leaked:
        return _ok("summary_mentions_none")
    return _fail("summary_mentions_none", f"summary mentions forbidden: {leaked}")


# ─── chart ───────────────────────────────────────────────────────────────

@predicate("chart_present")
def _chart_present(_: Any, state: dict) -> PredicateResult:
    if _chart_code(state):
        return _ok("chart_present")
    return _fail("chart_present", "no chart code generated")


@predicate("chart_absent")
def _chart_absent(_: Any, state: dict) -> PredicateResult:
    if not _chart_code(state):
        return _ok("chart_absent")
    return _fail("chart_absent", "chart code unexpectedly present")


@predicate("chart_code_contains")
def _chart_code_contains(needle: str, state: dict) -> PredicateResult:
    code = _chart_code(state) or ""
    if needle in code:
        return _ok("chart_code_contains")
    return _fail("chart_code_contains", f"chart code missing {needle!r}")


# ─── report ──────────────────────────────────────────────────────────────

@predicate("report_section_count_at_least")
def _report_count(n: int, state: dict) -> PredicateResult:
    sections = state.get("report_sections") or []
    if len(sections) >= n:
        return _ok("report_section_count_at_least")
    return _fail("report_section_count_at_least", f"expected ≥{n}, got {len(sections)}")


@predicate("report_all_sections_succeeded")
def _report_all_ok(_: Any, state: dict) -> PredicateResult:
    sections = state.get("report_sections") or []
    failed = [getattr(s, "title", "?") for s in sections if getattr(s, "error", None)]
    if not failed:
        return _ok("report_all_sections_succeeded")
    return _fail("report_all_sections_succeeded", f"failed sections: {failed}")


# ─── LLM judge ───────────────────────────────────────────────────────────
# Use sparingly — costs ~1 LLM call per check. Right tool when prose has many
# valid surface forms (e.g., "no Germany data" vs. "only SA/NZ have stores"
# both legitimately answer the same question).
#
# Judge runs on Haiku while the agent under test runs Sonnet for SQL/chart/
# planner — a different model class avoids the self-grading bias where a
# model rates its own outputs more leniently than independent ones.

class _JudgeVerdict(BaseModel):
    passed: bool = Field(description="Whether the summary satisfies the rubric.")
    reason: str = Field(description="One short sentence justifying the verdict.")


@cache
def _judge_llm():
    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
    return llm.with_structured_output(_JudgeVerdict)


_JUDGE_SYSTEM = """You evaluate whether an AI assistant's summary satisfies a rubric.

Read the rubric carefully. Multiple surface forms can satisfy the same rubric — \
the assistant might phrase the same correct answer many ways. Accept any phrasing \
that genuinely demonstrates the property the rubric asks for. Reject summaries \
that miss the property even if they're otherwise well-written.

Examples of acceptable equivalence:
- Rubric "acknowledges no data is available for Germany"
  ✓ "No Germany payments found."
  ✓ "Operational stores are only in South Africa and New Zealand, so Germany has no data."
  ✗ "Here is revenue by country: ..." (lists data without acknowledging Germany absence)

Be strict on substance, lenient on phrasing. Return passed=true/false plus one \
short reason sentence."""


@predicate("summary_satisfies")
def _summary_satisfies(rubric: str, state: dict) -> PredicateResult:
    """LLM-judge predicate. The arg is a natural-language rubric — what should
    be true of the summary. Right tool when the answer has many valid surface
    forms; substring checks are the right tool when there's a specific token
    you require (an entity name, a $ figure, etc.)."""
    summary = (state.get("summary") or "").strip()
    if not summary:
        return _fail("summary_satisfies", "summary is empty")
    user = f"Rubric:\n{rubric}\n\nSummary to judge:\n---\n{summary}\n---"
    try:
        verdict: _JudgeVerdict = _judge_llm().invoke(
            [{"role": "system", "content": _JUDGE_SYSTEM},
             {"role": "user", "content": user}]
        )
    except Exception as e:
        return _fail("summary_satisfies", f"judge LLM raised: {type(e).__name__}: {e}")
    if verdict.passed:
        return _ok("summary_satisfies")
    return _fail("summary_satisfies", f"judge: {verdict.reason}")
