"""Tests for the generate_chart node — LLM-driven chart code generation."""
from unittest.mock import MagicMock, patch

import pytest

from agent.nodes import generate_chart
from agent.state import ChartCode


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


# ---------- fast-path skips (no LLM call) ----------

def test_empty_rows_skips_llm():
    """No rows → no chart, no LLM call."""
    with patch("agent.nodes._chart_picker") as pick:
        out = generate_chart(_state(rows=[]))
    assert out == {"chart": None}
    pick.assert_not_called()


def test_single_row_skips_llm():
    """1 row → not chartable in auto mode, skip the LLM."""
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


def test_table_override_skips_llm():
    """User asked for a table view — no chart, no LLM call."""
    with patch("agent.nodes._chart_picker") as pick:
        out = generate_chart(_state(
            rows=[{"a": 1}, {"a": 2}],
            chart_kind_override="table",
        ))
    assert out == {"chart": None}
    pick.assert_not_called()


# ---------- LLM-driven generation ----------

def test_chartable_rows_call_llm_and_return_chart_code():
    rows = [{"title": f"film_{i}", "count": 30 - i} for i in range(5)]
    chart = ChartCode(
        reasoning="Bar chart compares discrete categories.",
        code="fig = px.bar(df, x='title', y='count', title='Top 5')",
        title="Top 5 Films by Rentals",
    )
    picker = MagicMock()
    picker.invoke.return_value = chart

    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows, data_question="Top 5 films"))

    assert out["chart"] is chart
    picker.invoke.assert_called_once()
    # Confirm the prompt actually carries the columns + sample rows
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "title" in user_content
    assert "count" in user_content
    assert "5 total" in user_content


def test_llm_returning_empty_code_means_no_chart():
    """LLM declines to plot (sets code=None) → no chart attached."""
    rows = [{"title": "x", "count": 1}, {"title": "y", "count": 2}]
    chart = ChartCode(reasoning="Single value, not useful to visualize.", code=None)
    picker = MagicMock()
    picker.invoke.return_value = chart
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))
    assert out == {"chart": None}


def test_llm_returning_blank_code_means_no_chart():
    """Empty-string code is treated the same as None — no chart."""
    rows = [{"title": "x", "count": 1}, {"title": "y", "count": 2}]
    chart = ChartCode(reasoning="empty", code="")
    picker = MagicMock()
    picker.invoke.return_value = chart
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))
    assert out == {"chart": None}
