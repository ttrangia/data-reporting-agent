"""Tests for the 'rechart' intent — front_agent skips SQL and routes
straight to generate_response → generate_chart with prior-turn rows."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import app_graph
from agent.nodes import generate_chart
from agent.schemas import FrontAgentDecision
from agent.state import ChartCode, turn_input


# ---------- schema validators ----------

def test_rechart_decision_valid():
    """FrontAgentDecision is now routing-only — no response_text field."""
    d = FrontAgentDecision(intent="rechart", chart_kind_override="pie")
    assert d.intent == "rechart"
    assert d.chart_kind_override == "pie"
    assert d.data_question is None


def test_rechart_decision_without_kind_override_is_allowed():
    """User asked 'replot' without specifying kind — let the picker decide."""
    d = FrontAgentDecision(intent="rechart")
    assert d.chart_kind_override is None


# ---------- generate_chart on rechart turns calls the picker with prior context ----------

def _state(**overrides):
    base = {
        "question": "anything",
        "data_question": "Top 5 films by 2022 rental count",
        "rows": [{"title": f"f{i}", "count": 30 - i} for i in range(5)],
        "sql_error": None,
        "intent": "rechart",
        "chart": ChartCode(code="fig = px.bar(df, x='title', y='count')", title="Prior caption"),
        "chart_kind_override": None,
        "retries": 0,
    }
    base.update(overrides)
    return base


def test_table_override_short_circuits_no_llm_call():
    """User wants table view — fast deterministic path, no LLM call."""
    with patch("agent.nodes._chart_picker") as picker:
        out = generate_chart(_state(chart_kind_override="table"))
    assert out == {"chart": None}
    picker.assert_not_called()


def test_rechart_calls_picker_with_prior_code_in_prompt():
    """The picker prompt MUST include the prior chart's code so the LLM has
    context to apply user modifications to the existing visualization."""
    prior = ChartCode(
        reasoning="Bar chart of top films",
        code="fig = px.bar(df, x='title', y='count', title='Top 5 films by 2022 rental count')",
        title="Top 5 films by 2022 rental count",
    )
    new_chart = ChartCode(
        reasoning="Pie chart variant",
        code="fig = px.pie(df, names='title', values='count', title='Top 5 films by 2022 rental count')",
        title="Top 5 films by 2022 rental count",
    )
    picker = MagicMock()
    picker.invoke.return_value = new_chart
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(
            question="Replot as a pie chart",
            chart=prior,
            chart_kind_override="pie",
        ))
    assert out["chart"] is new_chart
    picker.invoke.assert_called_once()
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    # Prior code AND the user's modification are visible to the picker
    assert "px.bar(df, x='title', y='count'" in user_content
    assert "Replot as a pie chart" in user_content


def test_rechart_axis_change_propagates_via_picker():
    """User-supplied 'switch the x axis to genre' reaches the picker, which
    can return code that uses the new x column."""
    rows = [
        {"genre": "Action", "title": "f1", "count": 10},
        {"genre": "Drama",  "title": "f2", "count": 8},
    ]
    prior = ChartCode(
        code="fig = px.bar(df, x='title', y='count', title='Top films per genre')",
        title="Top films per genre",
    )
    new_chart = ChartCode(
        code="fig = px.bar(df, x='genre', y='count', title='Counts by genre')",
        title="Counts by genre",
    )
    picker = MagicMock()
    picker.invoke.return_value = new_chart
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(
            question="Switch the x axis to genre",
            chart=prior,
            rows=rows,
            chart_kind_override=None,  # axis change, no kind change
        ))
    assert "x='genre'" in out["chart"].code
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "Switch the x axis to genre" in user_content


def test_rechart_picker_returning_no_code_falls_back_to_prior():
    """If the picker declines (code=None), preserve the user's existing chart."""
    prior = ChartCode(
        code="fig = px.bar(df, x='title', y='count')",
        title="Top films",
    )
    declined = ChartCode(reasoning="Couldn't figure it out", code=None)
    picker = MagicMock()
    picker.invoke.return_value = declined
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(
            question="Switch the x axis",
            chart=prior,
        ))
    assert out["chart"] is prior  # unchanged


def test_rechart_picker_dropping_to_none_with_no_prior_returns_none():
    """No prior chart AND picker declines → no chart at all."""
    declined = ChartCode(reasoning="Not chartable", code=None)
    picker = MagicMock()
    picker.invoke.return_value = declined
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(question="Try something else", chart=None))
    assert out == {"chart": None}


def test_data_turn_unaffected_by_rechart_logic():
    """Data turn (intent='data') uses the fresh CHART_CODE_USER prompt,
    not the rechart prompt. Prior chart is irrelevant."""
    chart = ChartCode(
        reasoning="Top 5",
        code="fig = px.bar(df, x='title', y='count')",
        title="Top 5 films",
    )
    picker = MagicMock()
    picker.invoke.return_value = chart
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(intent="data", question="Top 5 films"))
    assert out["chart"] is chart
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "MODIFY an existing chart" not in user_content  # rechart prompt not used
    assert "Question: Top 5 films" in user_content


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
        FrontAgentDecision(intent="rechart", chart_kind_override="pie"),
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

    # generate_response now runs on rechart turns — needs its own async mock.
    response_llm = MagicMock()
    response_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Here it is as a pie chart."))

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._sql_generator", return_value=sql_llm),
        patch("agent.nodes._summarizer", return_value=summary_llm),
        patch("agent.nodes._response_llm", return_value=response_llm),
        patch("agent.nodes._chart_picker") as chart_picker_mock,
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
        patch("agent.nodes.run_query", return_value=rows_returned),
    ):
        # Turn 1 chart picker returns the original bar; turn 2 returns the pie.
        chart_picker_mock.return_value.invoke.side_effect = [
            ChartCode(code="fig = px.bar(df, x='title', y='rentals')",
                      title="Top 5 films by rental count in 2022"),
            ChartCode(code="fig = px.pie(df, names='title', values='rentals')",
                      title="Top 5 films by rental count in 2022"),
        ]

        config = {"configurable": {"thread_id": "rechart-test"}}

        # Turn 1: data
        s1 = await app_graph.ainvoke(
            turn_input("Top 5 films", HumanMessage(content="Top 5 films")),
            config=config,
        )
        assert s1["intent"] == "data"
        assert s1["rows"] == rows_returned
        assert s1["chart"] is not None and "px.bar" in s1["chart"].code

        # Reset mocks to assert turn 2 invocations specifically
        sql_llm.invoke.reset_mock()
        chart_picker_mock.return_value.invoke.reset_mock()

        # Turn 2: rechart
        s2 = await app_graph.ainvoke(
            turn_input("Replot as a pie chart", HumanMessage(content="Replot as a pie chart")),
            config=config,
        )

    # SQL pipeline did NOT run on turn 2 (no new query)
    sql_llm.invoke.assert_not_called()
    # Chart picker IS called on turn 2 — the picker reads prior chart + user
    # request and applies the modification (pie kind, axes preserved).
    chart_picker_mock.return_value.invoke.assert_called_once()
    msgs = chart_picker_mock.return_value.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "MODIFY an existing chart" in user_content
    assert "Replot as a pie chart" in user_content
    # State carries the rechart intent and the new pie chart spec
    assert s2["intent"] == "rechart"
    assert s2["summary"] == "Here it is as a pie chart."
    assert s2["chart"].code is not None and "px.pie" in s2["chart"].code
    # Rows persisted from turn 1 (cross-turn memory)
    assert s2["rows"] == rows_returned
