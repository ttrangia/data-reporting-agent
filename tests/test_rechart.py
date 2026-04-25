"""Tests for the 'rechart' intent — front_agent skips SQL and routes
straight to generate_chart with prior-turn rows."""
from unittest.mock import AsyncMock, MagicMock, patch

import pydantic
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import app_graph
from agent.nodes import generate_chart
from agent.schemas import FrontAgentDecision
from agent.state import ChartSpec, turn_input


# ---------- schema validators ----------

def test_rechart_decision_valid():
    d = FrontAgentDecision(
        intent="rechart",
        response_text="Here it is as a pie chart.",
        chart_kind_override="pie",
    )
    assert d.intent == "rechart"
    assert d.chart_kind_override == "pie"
    assert d.data_question is None


def test_rechart_decision_without_response_text_rejected():
    with pytest.raises(pydantic.ValidationError):
        FrontAgentDecision(intent="rechart", chart_kind_override="pie")


def test_rechart_decision_without_kind_override_is_allowed():
    """User asked 'replot' without specifying kind — let the picker decide."""
    d = FrontAgentDecision(
        intent="rechart",
        response_text="Here's an updated chart.",
    )
    assert d.chart_kind_override is None


# ---------- generate_chart respects chart_kind_override ----------

def _state(**overrides):
    base = {
        "question": "anything",
        "data_question": "Top 5 films by 2022 rental count",
        "rows": [{"title": f"f{i}", "count": 30 - i} for i in range(5)],
        "sql_error": None,
        "chart_kind_override": None,
        "retries": 0,
    }
    base.update(overrides)
    return base


def test_override_pie_skips_llm_and_returns_pie_spec():
    with patch("agent.nodes._chart_picker") as picker:
        out = generate_chart(_state(chart_kind_override="pie"))
    assert out["chart"].kind == "pie"
    assert out["chart"].x in ("title", "count")
    assert out["chart"].y in ("title", "count")
    picker.assert_not_called()


def test_override_table_returns_no_chart():
    """User wants a table view — no Plotly figure should be produced."""
    with patch("agent.nodes._chart_picker") as picker:
        out = generate_chart(_state(chart_kind_override="table"))
    assert out == {"chart": None}
    picker.assert_not_called()


def test_override_line_skips_llm():
    with patch("agent.nodes._chart_picker") as picker:
        out = generate_chart(_state(chart_kind_override="line"))
    assert out["chart"].kind == "line"
    picker.assert_not_called()


def test_override_reuses_prior_chart_title_and_axes():
    """Regression: 'replot as pie' must NOT take 'replot as pie' as the
    chart title. It should reuse the prior chart's title (a real caption)
    and the prior x/y, just changing the kind."""
    prior = ChartSpec(
        kind="bar", x="title", y="rentals",
        title="Top 5 films by rental count, 2022",
        reasoning="(original turn)",
    )
    state = _state(
        question="Replot as a pie chart",  # current turn's literal text — must NOT become title
        chart_kind_override="pie",
        chart=prior,
    )
    with patch("agent.nodes._chart_picker") as picker:
        out = generate_chart(state)
    picker.assert_not_called()
    assert out["chart"].kind == "pie"
    assert out["chart"].x == "title"
    assert out["chart"].y == "rentals"
    assert out["chart"].title == "Top 5 films by rental count, 2022"
    # Sanity: the rechart command did NOT leak into the title
    assert "replot" not in out["chart"].title.lower()
    assert "pie chart" not in out["chart"].title.lower()


def test_override_with_no_prior_chart_uses_data_question_not_current_question():
    """If there's no prior chart object (e.g., the original chart was None),
    we still must NOT use the rechart command as the title. Fall back to the
    persisted data_question, which holds the original meaningful query."""
    state = _state(
        question="Replot as a pie chart",
        data_question="Top 5 films by rental count, 2022",
        chart=None,
        chart_kind_override="pie",
    )
    with patch("agent.nodes._chart_picker") as picker:
        out = generate_chart(state)
    picker.assert_not_called()
    assert "replot" not in out["chart"].title.lower()
    assert "Top 5 films" in out["chart"].title


def test_no_override_falls_through_to_llm():
    """Sanity: chart_kind_override=None → existing LLM-picker path runs."""
    spec = ChartSpec(kind="bar", x="title", y="count", title="ok")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(chart_kind_override=None))
    assert out["chart"] is spec
    picker.invoke.assert_called_once()


# ---------- end-to-end rechart through the graph ----------

@pytest.mark.asyncio
async def test_rechart_path_skips_sql_and_uses_persisted_rows():
    """Two-turn flow:
       Turn 1: data question → SQL pipeline → rows persist.
       Turn 2: 'replot as pie' → front_agent picks rechart →
                graph routes directly to generate_chart →
                generate_chart uses prior rows + override → pie spec."""
    front_llm = MagicMock()
    front_llm.invoke.side_effect = [
        FrontAgentDecision(intent="data", data_question="Top 5 films by rental count in 2022"),
        FrontAgentDecision(
            intent="rechart",
            response_text="Here it is as a pie chart.",
            chart_kind_override="pie",
        ),
    ]

    sql_llm = MagicMock()
    from agent.schemas import SQLGeneration
    sql_llm.invoke.return_value = SQLGeneration(
        reasoning="(test)", tables_used=["film", "rental"],
        sql="SELECT f.title, COUNT(*) AS rentals FROM film f LIMIT 5",
    )

    summary_llm = MagicMock()
    summary_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Top films:..."))

    rows_returned = [{"title": f"film_{i}", "rentals": 30 - i} for i in range(5)]

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._sql_generator", return_value=sql_llm),
        patch("agent.nodes._summarizer", return_value=summary_llm),
        patch("agent.nodes._chart_picker") as chart_picker_mock,  # SHOULD NOT BE CALLED on rechart turn
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.run_query", return_value=rows_returned),
    ):
        # generate_chart on turn 1 uses the LLM picker; mock its return.
        chart_picker_mock.return_value.invoke.return_value = ChartSpec(
            kind="bar", x="title", y="rentals", title="(turn 1 chart)"
        )

        config = {"configurable": {"thread_id": "rechart-test"}}

        # Turn 1: data
        s1 = await app_graph.ainvoke(
            turn_input("Top 5 films", HumanMessage(content="Top 5 films")),
            config=config,
        )
        assert s1["intent"] == "data"
        assert s1["rows"] == rows_returned
        assert s1["chart"] is not None and s1["chart"].kind == "bar"

        # Reset mocks to assert turn 2 invocations specifically
        sql_llm.invoke.reset_mock()
        chart_picker_mock.return_value.invoke.reset_mock()

        # Turn 2: rechart
        s2 = await app_graph.ainvoke(
            turn_input("Replot as a pie chart", HumanMessage(content="Replot as a pie chart")),
            config=config,
        )

    # SQL pipeline did NOT run on turn 2
    sql_llm.invoke.assert_not_called()
    # Chart LLM picker did NOT run (override path is deterministic)
    chart_picker_mock.return_value.invoke.assert_not_called()
    # State carries the rechart intent and the new pie chart spec
    assert s2["intent"] == "rechart"
    assert s2["summary"] == "Here it is as a pie chart."
    assert s2["chart"].kind == "pie"
    # Rows persisted from turn 1 (cross-turn memory)
    assert s2["rows"] == rows_returned
