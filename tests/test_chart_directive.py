"""Tests for chart-directive detection and its effect on generate_chart."""
from unittest.mock import MagicMock, patch

import pytest

from agent.chart_directive import detect
from agent.nodes import generate_chart
from agent.state import ChartSpec


# ---------- detector: skip patterns win over force ----------

@pytest.mark.parametrize("q", [
    "Top 5 films by rental count, no chart please",
    "Don't plot this",
    "Don't chart it",
    "Show me the data without a graph",
    "Just text, no figure",
    "Just numbers please",
    "Skip the chart",
    "No visualization needed",
    "I do not want a chart",
    "Text only",
])
def test_detect_skip(q):
    assert detect(q) == "skip"


@pytest.mark.parametrize("q", [
    "Plot top 5 films by rental count",
    "Chart this",
    "Graph the monthly revenue",
    "Visualize the rental trend",
    "Draw a chart of revenue per store",
    "Show me a bar chart of top films",
    "Show me as a line graph",
    "Make me a pie chart of revenue share",
    "Create a chart for this",
    "Give me a graph of monthly rentals",
    "Top films in a bar chart",
    "Monthly revenue as a line chart",
    "With a chart, please",
])
def test_detect_force(q):
    assert detect(q) == "force"


@pytest.mark.parametrize("q", [
    "Top 5 films by rental count",
    "How many active customers do we have?",
    "What's the busiest store?",
    "Hi",
    "",
    None,
])
def test_detect_auto(q):
    assert detect(q) == "auto"


# Make sure skip beats force when both signals are present.
def test_skip_beats_force_when_ambiguous():
    assert detect("Plot the top 5 — actually no, no chart please") == "skip"


# ---------- node behavior under each directive ----------

def _state(question: str, rows: list[dict]):
    return {
        "question": question,
        "data_question": question,
        "rows": rows,
        "sql_error": None,
        "retries": 0,
    }


def test_skip_directive_returns_none_without_calling_llm():
    rows = [{"title": f"f{i}", "n": 10 - i} for i in range(5)]
    with patch("agent.nodes._chart_picker") as picker:
        out = generate_chart(_state("Top 5 films, no chart", rows))
    assert out == {"chart": None}
    picker.assert_not_called()


def test_force_directive_overrides_llm_none():
    """User asked for a chart, LLM said 'none' — we must still produce a chart."""
    rows = [{"title": "a", "count": 1}, {"title": "b", "count": 2}]
    spec = ChartSpec(kind="none", reasoning="(LLM declined)")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state("Plot this", rows))
    assert out["chart"] is not None
    assert out["chart"].kind == "bar"
    assert out["chart"].y == "count"


def test_force_directive_passes_force_note_in_prompt():
    rows = [{"title": "a", "count": 1}, {"title": "b", "count": 2}]
    spec = ChartSpec(kind="bar", x="title", y="count", title="Counts")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        generate_chart(_state("Show me a bar chart of films", rows))
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "explicitly asked for a chart" in user_content
    assert 'MUST pick "bar"' in user_content


def test_auto_directive_no_force_note_in_prompt():
    rows = [{"title": "a", "count": 1}, {"title": "b", "count": 2}]
    spec = ChartSpec(kind="bar", x="title", y="count", title="Counts")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        generate_chart(_state("Top films", rows))
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "explicitly asked" not in user_content


def test_force_directive_overrides_single_row_skip():
    """auto-mode skips single-row results; force-mode still asks the LLM."""
    rows = [{"metric": "active_customers", "value": 599}]
    spec = ChartSpec(kind="bar", x="metric", y="value", title="One value")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state("Plot active customers", rows))
    picker.invoke.assert_called_once()
    assert out["chart"] is not None


def test_force_directive_with_invalid_columns_falls_back_to_default():
    rows = [{"title": "a", "count": 1}, {"title": "b", "count": 2}]
    spec = ChartSpec(kind="bar", x="bogus", y="alsobogus", title="Bad")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state("Plot this", rows))
    assert out["chart"] is not None
    # Default spec uses real columns
    assert out["chart"].x in ("title", "count")
    assert out["chart"].y in ("title", "count")


def test_auto_directive_with_invalid_columns_returns_none():
    """Same scenario as above but no force directive — should bail out."""
    rows = [{"title": "a", "count": 1}, {"title": "b", "count": 2}]
    spec = ChartSpec(kind="bar", x="bogus", y="alsobogus", title="Bad")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state("Top films", rows))
    assert out == {"chart": None}
