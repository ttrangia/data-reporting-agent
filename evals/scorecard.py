"""Markdown scorecard writer for an eval run.

The scorecard is the durable artifact: a single markdown file per run,
committed alongside the codebase so `git log evals/reports/` shows
quality drift over time. Sections are ordered so a reader can answer
in this sequence:
    1. "Did it pass?"  → top-line summary
    2. "Where did it fail?"  → per-case table + failure breakdown
    3. "Why did it fail?"  → per-case details with the actual state

Reports diff well in git when cases stay in dataset order.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path


def _summary_block(results, elapsed_s: float, sha: str) -> str:
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = 100.0 * passed / total if total else 0.0
    crashed = sum(1 for r in results if r.crashed)
    crash_note = f" · {crashed} crashed" if crashed else ""
    return f"**{passed}/{total} passed · {pct:.1f}%{crash_note} · {elapsed_s:.1f}s · git {sha}**"


def _per_case_table(results) -> str:
    rows = [
        "| id | status | time | failed predicates |",
        "|----|--------|------|-------------------|",
    ]
    for r in results:
        if r.crashed:
            errs = "; ".join(t.error for t in r.turn_results if t.error)
            rows.append(f"| `{r.case_id}` | 💥 | {r.elapsed_s:.1f}s | {errs} |")
            continue
        if r.passed:
            rows.append(f"| `{r.case_id}` | ✓ | {r.elapsed_s:.1f}s | |")
            continue
        failed = [p for p in r.predicate_results if not p.passed]
        details = "<br>".join(f"`{p.name}` — {p.reason}" for p in failed)
        rows.append(f"| `{r.case_id}` | ✗ | {r.elapsed_s:.1f}s | {details} |")
    return "\n".join(rows)


def _failure_breakdown(results) -> str:
    """Which predicates fail most often — points to the area to invest in."""
    counter: Counter = Counter()
    by_pred: dict[str, list[str]] = {}
    for r in results:
        for p in r.predicate_results:
            if not p.passed:
                counter[p.name] += 1
                by_pred.setdefault(p.name, []).append(r.case_id)
    if not counter:
        return "_No failed predicates._"
    lines = ["| predicate | count | cases |", "|-----------|-------|-------|"]
    for name, n in counter.most_common():
        cases = ", ".join(f"`{c}`" for c in by_pred[name])
        lines.append(f"| `{name}` | {n} | {cases} |")
    return "\n".join(lines)


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"


def _per_turn_details(results) -> str:
    """Failed cases get an unfolded view: question, intent, SQL preview,
    rows count, summary preview. Saves the reader from re-running."""
    chunks = []
    for r in results:
        if r.passed:
            continue
        chunks.append(f"### `{r.case_id}`")
        if r.notes:
            chunks.append(f"_{r.notes.strip()}_")
        for i, t in enumerate(r.turn_results, 1):
            label = f"**Turn {i}**" if len(r.turn_results) > 1 else "**Q**"
            chunks.append(f"{label}: {t.question}")
            if t.error:
                chunks.append(f"- 💥 **error:** `{t.error}`")
                continue
            intent = t.state.get("intent") or "(none)"
            chunks.append(f"- intent: `{intent}`")
            sql = (t.state.get("sql") or "").strip()
            if sql:
                chunks.append(f"- sql: `{_truncate(sql, 240)}`")
            rows = t.state.get("rows")
            if rows is not None:
                chunks.append(f"- rows: {len(rows)}")
            summary = (t.state.get("summary") or "").strip()
            if summary:
                chunks.append(f"- summary: _{_truncate(summary, 200)}_")
            sections = t.state.get("report_sections") or []
            if sections:
                titles = [getattr(s, "title", "?") for s in sections]
                chunks.append(f"- report sections: {titles}")
        chunks.append("")
    return "\n".join(chunks) if chunks else "_All cases passed!_"


def write_scorecard(path: Path, results, total_elapsed_s: float, git_sha: str) -> None:
    parts = [
        "# Eval Scorecard",
        "",
        _summary_block(results, total_elapsed_s, git_sha),
        "",
        "## Per-case results",
        "",
        _per_case_table(results),
        "",
        "## Failures by predicate",
        "",
        _failure_breakdown(results),
        "",
        "## Failed cases — details",
        "",
        _per_turn_details(results),
        "",
    ]
    path.write_text("\n".join(parts))
