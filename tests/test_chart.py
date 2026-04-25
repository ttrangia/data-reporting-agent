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


def test_group_column_validated_against_real_columns():
    """If the LLM puts a bogus group column in the spec, drop it silently
    rather than failing the chart entirely. The 2D chart is still useful."""
    rows = [{"genre": "Action", "title": "x", "count": 1},
            {"genre": "Drama", "title": "y", "count": 2}]
    spec = ChartSpec(kind="bar", x="title", y="count", group="bogus_col",
                     title="Grouped attempt")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))
    assert out["chart"].kind == "bar"
    assert out["chart"].group is None  # bogus group dropped


def test_pie_kind_strips_group_column():
    """Pie charts don't support color grouping. Drop any group the LLM emits."""
    rows = [{"store": "A", "revenue": 100},
            {"store": "B", "revenue": 200}]
    spec = ChartSpec(kind="pie", x="store", y="revenue", group="store",
                     title="Revenue share")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))
    assert out["chart"].kind == "pie"
    assert out["chart"].group is None


def test_valid_group_column_preserved():
    """When group is a real column on a bar chart, keep it for the renderer."""
    rows = [{"genre": "Action", "title": "x", "count": 1},
            {"genre": "Drama", "title": "y", "count": 2}]
    spec = ChartSpec(kind="bar", x="title", y="count", group="genre",
                     title="Top films per genre")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))
    assert out["chart"].group == "genre"


def test_layout_knobs_validated_against_columns():
    """facet_col and sort_by must be real columns; bogus values dropped."""
    rows = [{"genre": "Action", "title": "x", "count": 1},
            {"genre": "Drama", "title": "y", "count": 2}]
    spec = ChartSpec(
        kind="bar", x="title", y="count", group="genre",
        title="Test",
        facet_col="bogus_facet", sort_by="bogus_sort", sort_desc=True,
        barmode="stack", orientation="h",
    )
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))
    assert out["chart"].facet_col is None  # dropped
    assert out["chart"].sort_by is None    # dropped
    assert out["chart"].sort_desc is None  # dropped along with sort_by
    # Valid layout knobs preserved
    assert out["chart"].barmode == "stack"
    assert out["chart"].orientation == "h"


def test_layout_knobs_stripped_from_pie():
    """Pie shouldn't carry barmode/orientation/facet/sort/group."""
    rows = [{"store": "A", "revenue": 100}, {"store": "B", "revenue": 200}]
    spec = ChartSpec(
        kind="pie", x="store", y="revenue",
        title="Share",
        group="store", barmode="group", orientation="h",
        facet_col="store", sort_by="revenue", sort_desc=True,
    )
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))
    assert out["chart"].kind == "pie"
    assert out["chart"].group is None
    assert out["chart"].barmode is None
    assert out["chart"].orientation is None
    assert out["chart"].facet_col is None
    assert out["chart"].sort_by is None


def test_layout_knobs_partially_valid_partially_dropped():
    """Mixed: facet_col valid, sort_by invalid → keep facet, drop sort."""
    rows = [{"genre": "Action", "title": "x", "count": 1}]
    rows.extend([{"genre": "Drama", "title": "y", "count": 2}])
    spec = ChartSpec(
        kind="bar", x="title", y="count",
        title="t",
        facet_col="genre",      # valid
        sort_by="not_a_col",    # invalid
        sort_desc=True,
    )
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(rows=rows))
    assert out["chart"].facet_col == "genre"
    assert out["chart"].sort_by is None


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
