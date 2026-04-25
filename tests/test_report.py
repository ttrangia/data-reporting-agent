"""Tests for the report path — front_agent routes broad asks to plan_report,
which fans out parallel sub_query branches that aggregate_report combines."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import app_graph
from agent.nodes import _section_chart, plan_report, sub_query
from agent.schemas import FrontAgentDecision, ReportPlan, ReportSection, SQLGeneration
from agent.state import ChartCode, turn_input


# ---------- _section_chart helper ----------

def test_section_chart_skips_when_chart_hint_is_none():
    sec = ReportSection(title="x", sub_question="x", chart_hint="none")
    # No LLM call should happen for hint="none"
    with patch("agent.nodes._chart_picker") as picker:
        out = _section_chart(sec, [{"a": 1, "b": 2}])
    assert out is None
    picker.assert_not_called()


def test_section_chart_skips_when_no_rows():
    sec = ReportSection(title="x", sub_question="x", chart_hint="bar")
    with patch("agent.nodes._chart_picker") as picker:
        out = _section_chart(sec, [])
    assert out is None
    picker.assert_not_called()


def test_section_chart_calls_llm_with_hint_and_section_context():
    """The chart-coder gets the section's sub_question + chart_hint baked in."""
    sec = ReportSection(title="Top films", sub_question="Top 5 films by rentals",
                        chart_hint="bar")
    rows = [{"title": "f1", "n": 10}, {"title": "f2", "n": 8}]
    chart = ChartCode(reasoning="Top 5 by rentals",
                      code="fig = px.bar(df, x='title', y='n')",
                      title="Top 5 Films by Rentals")
    picker = MagicMock()
    picker.invoke.return_value = chart
    with patch("agent.nodes._chart_picker", return_value=picker):
        out = _section_chart(sec, rows)
    assert out is chart
    msgs = picker.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert "Top 5 films by rentals" in user_content
    assert "Top films" in user_content   # title surfaced as section context
    assert "bar" in user_content          # chart_hint surfaced
    # Force note is included since report sections opted-in to a chart
    assert "explicitly asked for a chart" in user_content


# ---------- plan_report node ----------

def test_plan_report_resets_accumulator_and_returns_outline():
    plan = ReportPlan(
        sections=[
            ReportSection(title="Headline", sub_question="Total revenue 2022", chart_hint="none"),
            ReportSection(title="Trend", sub_question="Revenue per month", chart_hint="line"),
            ReportSection(title="Top films", sub_question="Top 10 films by rentals", chart_hint="bar"),
        ],
        rationale="Quarterly report combining headline, trend, and top performers.",
    )
    planner = MagicMock()
    planner.invoke.return_value = plan

    with (
        patch("agent.nodes._report_planner", return_value=planner),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
    ):
        out = plan_report({
            "question": "create a quarterly report",
            "data_question": None,
            "messages": [],
        })

    assert out["report_outline"] == plan.sections
    assert "Quarterly report" in out["report_plan_rationale"]
    # Reset sentinel — None tells the custom reducer to clear the accumulator
    assert out["report_sections"] is None


# ---------- sub_query node ----------

@pytest.mark.asyncio
async def test_sub_query_completes_a_section():
    section = ReportSection(
        title="Top 5 films",
        sub_question="Top 5 films by rental count in 2022",
        chart_hint="bar",
    )
    sql_gen = MagicMock()
    sql_gen.invoke.return_value = SQLGeneration(
        reasoning="x", tables_used=["film"],
        sql="SELECT title, count FROM film LIMIT 5",
    )
    summarizer = MagicMock()
    summarizer.ainvoke = AsyncMock(return_value=AIMessage(content="Top film led with 34 rentals."))
    chart_picker = MagicMock()
    chart_picker.invoke.return_value = ChartCode(
        reasoning="bar of top films",
        code="fig = px.bar(df, x='title', y='count')",
        title="Top 5 Films",
    )

    with (
        patch("agent.nodes._sql_generator", return_value=sql_gen),
        patch("agent.nodes._section_summarizer", return_value=summarizer),
        patch("agent.nodes._chart_picker", return_value=chart_picker),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
        patch("agent.nodes.run_query", return_value=[
            {"title": "BUCKET", "count": 34},
            {"title": "ROCKETEER", "count": 31},
        ]),
    ):
        out = await sub_query({
            "current_section": section,
            "question": "any",
        })

    sections = out["report_sections"]
    assert len(sections) == 1
    s = sections[0]
    assert s.title == "Top 5 films"
    assert s.row_count == 2
    assert s.summary == "Top film led with 34 rentals."
    assert s.error is None
    assert s.chart is not None and s.chart.code is not None
    assert "px.bar" in s.chart.code


@pytest.mark.asyncio
async def test_sub_query_degrades_gracefully_on_sql_error():
    """SQL generation failure → section keeps its title but gets an error."""
    section = ReportSection(title="Bad", sub_question="x", chart_hint="bar")
    sql_gen = MagicMock()
    sql_gen.invoke.side_effect = RuntimeError("API timeout")

    with (
        patch("agent.nodes._sql_generator", return_value=sql_gen),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
    ):
        out = await sub_query({"current_section": section, "question": "any"})

    s = out["report_sections"][0]
    assert s.error is not None
    assert "SQL generation failed" in s.error
    assert s.title == "Bad"  # title preserved


@pytest.mark.asyncio
async def test_sub_query_graceful_on_guard_rejection():
    section = ReportSection(title="Mut", sub_question="x", chart_hint="bar")
    sql_gen = MagicMock()
    sql_gen.invoke.return_value = SQLGeneration(
        reasoning="x", tables_used=["customer"], sql="DELETE FROM customer",
    )

    with (
        patch("agent.nodes._sql_generator", return_value=sql_gen),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
    ):
        out = await sub_query({"current_section": section, "question": "any"})

    s = out["report_sections"][0]
    assert s.error is not None
    assert "Validator rejected" in s.error


# ---------- end-to-end through the graph ----------

@pytest.mark.asyncio
async def test_report_flow_runs_parallel_sub_queries_and_aggregates():
    """Full graph: front_agent picks 'report' → plan_report decomposes →
    fan-out to parallel sub_query × N → aggregate_report composes the report.

    Asserts: SQL gen ran once per section (not for the user's broad question
    directly), all sections appear in state.report_sections, and the
    aggregator was called with all sections in its prompt."""
    front_llm = MagicMock()
    front_llm.invoke.return_value = FrontAgentDecision(intent="report")

    plan = ReportPlan(
        sections=[
            ReportSection(title="Headline", sub_question="Total revenue 2022", chart_hint="none"),
            ReportSection(title="Trend", sub_question="Revenue per month 2022", chart_hint="line"),
            ReportSection(title="Top films", sub_question="Top 10 films by rentals 2022", chart_hint="bar"),
        ],
        rationale="Quarterly business overview.",
    )
    planner = MagicMock()
    planner.invoke.return_value = plan

    # Each section produces a different SQLGeneration; all share the same SQL gen mock
    sql_gen = MagicMock()
    sql_gen.invoke.side_effect = [
        SQLGeneration(reasoning="x", tables_used=["payment"], sql="SELECT SUM(amount) AS total FROM payment LIMIT 1"),
        SQLGeneration(reasoning="x", tables_used=["payment"], sql="SELECT date_trunc('month', payment_date) AS m, SUM(amount) AS r FROM payment GROUP BY m LIMIT 12"),
        SQLGeneration(reasoning="x", tables_used=["film", "rental"], sql="SELECT title, COUNT(*) AS n FROM film f JOIN rental r ON True LIMIT 10"),
    ]

    rows_per_query = [
        [{"total": 67000}],
        [{"m": "2022-02", "r": 5000}, {"m": "2022-03", "r": 8000}],
        [{"title": "f1", "n": 34}, {"title": "f2", "n": 31}],
    ]

    def fake_run_query(sql):
        # Each call peels one row set off the queue
        return rows_per_query.pop(0) if rows_per_query else []

    summarizer = MagicMock()
    summarizer.ainvoke = AsyncMock(return_value=AIMessage(content="(section blurb)"))

    aggregator = MagicMock()
    aggregator.ainvoke = AsyncMock(return_value=AIMessage(
        content="Revenue totaled $67,000 across the 2022 period. Trend showed growth from Feb to Mar."
    ))
    # sub_query now calls the chart picker per section (chart_hint != "none").
    chart_picker = MagicMock()
    chart_picker.invoke.return_value = ChartCode(
        reasoning="ok", code="fig = px.bar(df, x=df.columns[0], y=df.columns[1])",
        title="Section chart",
    )

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._report_planner", return_value=planner),
        patch("agent.nodes._sql_generator", return_value=sql_gen),
        patch("agent.nodes._section_summarizer", return_value=summarizer),
        patch("agent.nodes._chart_picker", return_value=chart_picker),
        patch("agent.nodes._report_aggregator", return_value=aggregator),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
        patch("agent.nodes.run_query", side_effect=fake_run_query),
    ):
        config = {"configurable": {"thread_id": "report-test"}}
        final = await app_graph.ainvoke(
            turn_input("create a quarterly sales report",
                       HumanMessage(content="create a quarterly sales report")),
            config=config,
        )

    # front_agent picked report
    assert final["intent"] == "report"
    # Planner ran once
    planner.invoke.assert_called_once()
    # 3 sections were planned
    assert len(final["report_outline"]) == 3
    # 3 SQL gens fired (one per section, NOT one for the broad question)
    assert sql_gen.invoke.call_count == 3
    # All 3 sections completed
    assert len(final["report_sections"]) == 3
    # Aggregator ran once and produced the final text
    aggregator.ainvoke.assert_called_once()
    assert "67,000" in final["summary"]
    assert "67,000" in final["report_text"]
    # Aggregator's prompt included all section blurbs
    aggregator_user = next(
        m for m in aggregator.ainvoke.call_args.args[0] if m["role"] == "user"
    )["content"]
    assert "Headline" in aggregator_user
    assert "Trend" in aggregator_user
    assert "Top films" in aggregator_user


@pytest.mark.asyncio
async def test_report_flow_continues_when_one_section_fails():
    """One section's SQL fails → others still complete → aggregator gets
    partial results + a 'failed sections' note."""
    front_llm = MagicMock()
    front_llm.invoke.return_value = FrontAgentDecision(intent="report")

    plan = ReportPlan(
        sections=[
            ReportSection(title="Good", sub_question="Total revenue", chart_hint="none"),
            ReportSection(title="Bad", sub_question="Bogus", chart_hint="bar"),
        ],
        rationale="x",
    )
    planner = MagicMock()
    planner.invoke.return_value = plan

    sql_gen = MagicMock()
    sql_gen.invoke.side_effect = [
        SQLGeneration(reasoning="x", tables_used=["payment"], sql="SELECT 1 AS total"),
        # Second section's SQL is mutation → guard rejects
        SQLGeneration(reasoning="x", tables_used=["customer"], sql="DELETE FROM customer"),
    ]

    summarizer = MagicMock()
    summarizer.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    aggregator = MagicMock()
    aggregator.ainvoke = AsyncMock(return_value=AIMessage(content="Partial report."))
    chart_picker = MagicMock()
    chart_picker.invoke.return_value = ChartCode(
        reasoning="ok", code="fig = None", title="(no chart)",
    )

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._report_planner", return_value=planner),
        patch("agent.nodes._sql_generator", return_value=sql_gen),
        patch("agent.nodes._section_summarizer", return_value=summarizer),
        patch("agent.nodes._chart_picker", return_value=chart_picker),
        patch("agent.nodes._report_aggregator", return_value=aggregator),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
        patch("agent.nodes.run_query", return_value=[{"total": 100}]),
    ):
        config = {"configurable": {"thread_id": "report-partial"}}
        final = await app_graph.ainvoke(
            turn_input("report", HumanMessage(content="report")),
            config=config,
        )

    # Both sections present; one has error, one is clean
    sections = final["report_sections"]
    assert len(sections) == 2
    by_title = {s.title: s for s in sections}
    assert by_title["Good"].error is None
    assert by_title["Bad"].error is not None
    # Aggregator still ran
    aggregator.ainvoke.assert_called_once()
    # Failed section appears in the aggregator prompt
    aggregator_user = next(
        m for m in aggregator.ainvoke.call_args.args[0] if m["role"] == "user"
    )["content"]
    assert "Bad" in aggregator_user
