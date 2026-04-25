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


# ---------- generate_chart on rechart turns calls the picker with prior context ----------

def _state(**overrides):
    base = {
        "question": "anything",
        "data_question": "Top 5 films by 2022 rental count",
        "rows": [{"title": f"f{i}", "count": 30 - i} for i in range(5)],
        "sql_error": None,
        "intent": "rechart",
        "chart": ChartSpec(kind="bar", x="title", y="count", title="Prior caption"),
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


def test_rechart_calls_picker_with_prior_chart_in_prompt():
    """The picker prompt MUST include the prior chart spec so the LLM has
    context to apply user modifications."""
    prior = ChartSpec(kind="bar", x="title", y="count",
                      title="Top 5 films by 2022 rental count", group=None)
    new_spec = ChartSpec(kind="pie", x="title", y="count",
                         title="Top 5 films by 2022 rental count")
    picker = MagicMock()
    picker.invoke.return_value = new_spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(
            question="Replot as a pie chart",
            chart=prior,
            chart_kind_override="pie",
        ))
    assert out["chart"] is new_spec
    picker.invoke.assert_called_once()
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    # Prior chart shape and the user's modification are both visible to the picker
    assert "Top 5 films by 2022 rental count" in user_content
    assert "Replot as a pie chart" in user_content
    import re
    assert re.search(r"kind:\s+bar", user_content), "prior kind not visible to picker"


def test_rechart_axis_change_propagates_via_picker():
    """The user-supplied phrasing 'switch the x axis to genre' should reach
    the picker. The picker can then return a spec with the new x."""
    rows = [
        {"genre": "Action", "title": "f1", "count": 10},
        {"genre": "Drama",  "title": "f2", "count": 8},
    ]
    prior = ChartSpec(kind="bar", x="title", y="count",
                      title="Top films per genre", group=None)
    new_spec = ChartSpec(kind="bar", x="genre", y="count",
                         title="Counts by genre", group=None)
    picker = MagicMock()
    picker.invoke.return_value = new_spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(
            question="Switch the x axis to genre",
            chart=prior,
            rows=rows,
            chart_kind_override=None,  # axis change, no kind change
        ))
    assert out["chart"].x == "genre"
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "Switch the x axis to genre" in user_content


def test_rechart_invalid_columns_fall_back_to_prior():
    """If the picker hallucinates a column, don't erase the user's chart —
    keep the prior one rather than returning None."""
    prior = ChartSpec(kind="bar", x="title", y="count",
                      title="Top films", group=None)
    new_spec = ChartSpec(kind="bar", x="bogus", y="count", title="Bad")
    picker = MagicMock()
    picker.invoke.return_value = new_spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(
            question="Switch the x axis",
            chart=prior,
        ))
    assert out["chart"] is prior  # unchanged, not erased


def test_rechart_picker_dropping_to_none_returns_no_chart():
    """If the picker decides the result isn't chartable, return no chart."""
    new_spec = ChartSpec(kind="none", reasoning="(actually unchartable)")
    picker = MagicMock()
    picker.invoke.return_value = new_spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(question="Try something else"))
    assert out == {"chart": None}


def test_data_turn_unaffected_by_rechart_logic():
    """Sanity: data turn (intent="data") still uses the regular CHART_SPEC_USER
    template, NOT the rechart prompt. Prior chart is irrelevant."""
    spec = ChartSpec(kind="bar", x="title", y="count", title="ok")
    picker = MagicMock()
    picker.invoke.return_value = spec
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = generate_chart(_state(intent="data", question="Top 5 films"))
    assert out["chart"] is spec
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "modify an EXISTING chart" not in user_content  # rechart prompt not used
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
        patch("agent.nodes._chart_picker") as chart_picker_mock,
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.run_query", return_value=rows_returned),
    ):
        # Turn 1 chart picker returns the original bar; turn 2 returns the pie.
        chart_picker_mock.return_value.invoke.side_effect = [
            ChartSpec(kind="bar", x="title", y="rentals", title="Top 5 films by rental count in 2022"),
            ChartSpec(kind="pie", x="title", y="rentals", title="Top 5 films by rental count in 2022"),
        ]

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

    # SQL pipeline did NOT run on turn 2 (no new query)
    sql_llm.invoke.assert_not_called()
    # Chart picker IS called on turn 2 — the picker reads prior chart + user
    # request and applies the modification (pie kind, axes preserved).
    chart_picker_mock.return_value.invoke.assert_called_once()
    msgs = chart_picker_mock.return_value.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "modify an EXISTING chart" in user_content
    assert "Replot as a pie chart" in user_content
    # State carries the rechart intent and the new pie chart spec
    assert s2["intent"] == "rechart"
    assert s2["summary"] == "Here it is as a pie chart."
    assert s2["chart"].kind == "pie"
    # Rows persisted from turn 1 (cross-turn memory)
    assert s2["rows"] == rows_returned
