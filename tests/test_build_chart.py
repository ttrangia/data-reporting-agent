"""Tests for app._build_chart — Plotly figure construction from ChartSpec.
Builds standalone (Chainlit context not required for plotly.express.Figure)."""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test")  # app.py imports warmup; doesn't fire it standalone

import pytest

# Importing app.py runs warmup() which hits Neon. Bypass by importing the
# function directly via module reload with warmup mocked.
from unittest.mock import patch

with patch("agent.db.warmup", lambda: None):
    from app import _build_chart

from agent.state import ChartSpec


ROWS_2D = [{"title": f"f{i}", "count": 10 - i} for i in range(5)]
ROWS_3D = [
    {"genre": "Action", "title": "f1", "count": 10},
    {"genre": "Action", "title": "f2", "count": 8},
    {"genre": "Drama",  "title": "f3", "count": 12},
    {"genre": "Drama",  "title": "f4", "count": 5},
]


def test_bar_2d_no_group():
    fig = _build_chart(ChartSpec(kind="bar", x="title", y="count", title="2D"), ROWS_2D)
    assert fig is not None
    assert fig.layout.title.text == "2D"
    # No color dimension on 2D
    traces = fig.data
    assert len(traces) == 1


def test_bar_with_group_produces_grouped_chart():
    spec = ChartSpec(kind="bar", x="title", y="count", group="genre",
                     title="Top films per genre")
    fig = _build_chart(spec, ROWS_3D)
    assert fig is not None
    assert fig.layout.title.text == "Top films per genre"
    # Plotly creates one trace per group value when color= is set
    trace_names = {t.name for t in fig.data}
    assert "Action" in trace_names
    assert "Drama" in trace_names


def test_line_with_group_produces_multi_series():
    rows = [
        {"month": "Jan", "revenue": 100, "store": "A"},
        {"month": "Feb", "revenue": 120, "store": "A"},
        {"month": "Jan", "revenue": 80,  "store": "B"},
        {"month": "Feb", "revenue": 90,  "store": "B"},
    ]
    spec = ChartSpec(kind="line", x="month", y="revenue", group="store",
                     title="Monthly revenue by store")
    fig = _build_chart(spec, rows)
    assert fig is not None
    trace_names = {t.name for t in fig.data}
    assert {"A", "B"} <= trace_names


def test_pie_ignores_group_field():
    """Even if group somehow leaked through, pie should render as a single pie."""
    rows = [{"store": "A", "revenue": 100}, {"store": "B", "revenue": 200}]
    spec = ChartSpec(kind="pie", x="store", y="revenue", group="store", title="Share")
    fig = _build_chart(spec, rows)
    assert fig is not None
    # Pie has one trace (the slices), color in pie doesn't create additional traces
    assert len(fig.data) == 1


def test_kind_none_returns_none():
    spec = ChartSpec(kind="none")
    assert _build_chart(spec, ROWS_2D) is None


def test_kind_table_returns_none():
    spec = ChartSpec(kind="table")
    assert _build_chart(spec, ROWS_2D) is None


def test_empty_rows_returns_none():
    spec = ChartSpec(kind="bar", x="title", y="count", title="x")
    assert _build_chart(spec, []) is None


def test_facet_col_creates_subplots():
    """facet_col=genre should produce one subplot per genre value."""
    rows = [
        {"genre": "Action", "title": "f1", "count": 10},
        {"genre": "Action", "title": "f2", "count": 8},
        {"genre": "Drama",  "title": "f3", "count": 12},
        {"genre": "Drama",  "title": "f4", "count": 5},
        {"genre": "Comedy", "title": "f5", "count": 7},
    ]
    spec = ChartSpec(kind="bar", x="title", y="count", facet_col="genre",
                     title="Top per genre")
    fig = _build_chart(spec, rows)
    assert fig is not None
    # Plotly creates annotations (one per facet) for facet titles
    annotation_texts = [a.text for a in fig.layout.annotations]
    assert any("Action" in t for t in annotation_texts)
    assert any("Drama" in t for t in annotation_texts)
    assert any("Comedy" in t for t in annotation_texts)


def test_horizontal_orientation_flips_axes():
    rows = [{"genre": "Action", "count": 10}, {"genre": "Drama", "count": 12}]
    spec = ChartSpec(kind="bar", x="count", y="genre", orientation="h",
                     title="Horizontal")
    fig = _build_chart(spec, rows)
    assert fig is not None
    assert fig.data[0].orientation == "h"


def test_barmode_stack():
    rows = [
        {"month": "Jan", "store": "A", "revenue": 100},
        {"month": "Jan", "store": "B", "revenue": 80},
        {"month": "Feb", "store": "A", "revenue": 120},
        {"month": "Feb", "store": "B", "revenue": 90},
    ]
    spec = ChartSpec(kind="bar", x="month", y="revenue", group="store",
                     barmode="stack", title="Stacked")
    fig = _build_chart(spec, rows)
    assert fig is not None
    assert fig.layout.barmode == "stack"


def test_sort_by_count_descending():
    """Sorting should reorder the dataframe before plotly sees it."""
    rows = [{"name": "a", "v": 5}, {"name": "b", "v": 50}, {"name": "c", "v": 25}]
    spec = ChartSpec(kind="bar", x="name", y="v", sort_by="v", sort_desc=True,
                     title="Sorted")
    fig = _build_chart(spec, rows)
    assert fig is not None
    # First trace's x values reflect the sorted order: highest v first
    assert list(fig.data[0].x) == ["b", "c", "a"]
