"""Tests for the generate_chart node — visualization picker after summarize."""
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes import generate_chart
from agent.state import ChartSpec


def _state(**overrides) -> dict:
    base = {
        "question": "anything",
        "data_question": "anything",
        "rows": [],
        "sql_error": None,
        "retries": 0,
    }
    base.update(overrides)
    return base


def test_empty_rows_skips_llm():
    """No rows → no chart, no LLM call."""
    with patch("agent.nodes._chart_picker") as pick:
        out = generate_chart(_state(rows=[]))
    assert out == {"chart": None}
    pick.assert_not_called()


def test_single_row_skips_llm():
    """1 row → not chartable, skip the LLM."""
    with patch("agent.nodes._chart_picker") as pick:
        out = generate_chart(_state(rows=[{"count": 5}]))
    assert out == {"chart": None}
    pick.assert_not_called()


def test_sql_error_skips_llm():
    """Degraded path — execute_sql failed, no rows worth charting."""
    with patch("agent.nodes._chart_picker") as pick:
        out = generate_chart(_state(rows=[], sql_error="anything"))
    assert out == {"chart": None}
    pick.assert_not_called()


def test_chartable_rows_call_llm_and_return_spec():
    rows = [{"title": f"film_{i}", "count": 30 - i} for i in range(5)]
    spec = ChartSpec(kind="bar", x="title", y="count",
                     title="Top 5 films by rentals",
                     reasoning="Bar chart compares discrete categories.")
    picker = MagicMock()
    picker.invoke.return_value = spec

    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows, data_question="Top 5 films"))

    assert out["chart"] is spec
    picker.invoke.assert_called_once()
    # Confirm the prompt actually carries the columns + sample rows
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "title" in user_content
    assert "count" in user_content
    assert "5 total" in user_content


def test_llm_returning_none_kind_means_no_chart():
    rows = [{"title": "x", "count": 1}, {"title": "y", "count": 2}]
    spec = ChartSpec(kind="none", reasoning="Not useful to visualize.")
    picker = MagicMock()
    picker.invoke.return_value = spec

    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))

    assert out == {"chart": None}


def test_invalid_column_choice_falls_back_to_none():
    """Defense-in-depth: LLM picks a column that isn't in the result set →
    drop the spec rather than crashing the renderer downstream."""
    rows = [{"title": "x", "count": 1}, {"title": "y", "count": 2}]
    spec = ChartSpec(kind="bar", x="bogus_col", y="count", title="Bad")
    picker = MagicMock()
    picker.invoke.return_value = spec

    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))

    assert out == {"chart": None}


def test_table_kind_passes_through():
    """'table' is a valid choice (no chart needed) — distinct from 'none'.
    We propagate it so the UI could distinguish 'rendered as a table' from
    'no visualization applicable', even if today both render the same way."""
    rows = [{"name": "a", "email": "x@y"}, {"name": "b", "email": "y@z"}]
    spec = ChartSpec(kind="table", reasoning="Best read as a table.")
    picker = MagicMock()
    picker.invoke.return_value = spec

    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))

    # Per current design, "table" returns the spec (caller chooses what to do)
    assert out["chart"] is spec
    assert out["chart"].kind == "table"
