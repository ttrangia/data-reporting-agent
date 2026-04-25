"""Tests for diagnose_empty — runs a follow-up query when execute_sql returns
zero rows, surfacing what values DO have data so summarize can suggest
alternatives instead of just saying 'no rows'."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import app_graph
from agent.nodes import diagnose_empty
from agent.schemas import FrontAgentDecision, SQLGeneration
from agent.state import ChartSpec, turn_input


# ---------- node-level: short-circuits ----------

def test_skips_when_sql_error_present():
    """sql_error set means we got here via the retry path; not a 'real empty'."""
    with patch("agent.nodes._sql_generator") as gen:
        out = diagnose_empty({
            "question": "any", "data_question": "any",
            "sql": "SELECT 1", "sql_error": "boom",
            "rows": [], "retries": 0,
        })
    assert out == {"diagnostic_sql": None, "diagnostic_rows": None}
    gen.assert_not_called()


def test_skips_when_rows_already_present():
    """If rows is non-empty, no need to diagnose."""
    with patch("agent.nodes._sql_generator") as gen:
        out = diagnose_empty({
            "question": "any", "data_question": "any",
            "sql": "SELECT 1", "sql_error": None,
            "rows": [{"x": 1}], "retries": 0,
        })
    assert out == {"diagnostic_sql": None, "diagnostic_rows": None}
    gen.assert_not_called()


# ---------- node-level: happy path ----------

def test_runs_diagnostic_on_empty_result():
    """Empty rows → call SQL gen with diagnostic prompt → run via run_query."""
    diagnostic_sql_returned = SQLGeneration(
        reasoning="(test)", tables_used=["country", "store"],
        sql=("SELECT co.country, COUNT(*) AS n FROM store s "
             "JOIN address a ON s.address_id = a.address_id "
             "JOIN city ci ON a.city_id = ci.city_id "
             "JOIN country co ON ci.country_id = co.country_id "
             "GROUP BY co.country ORDER BY n DESC LIMIT 20"),
    )
    diagnostic_rows = [
        {"country": "Australia", "n": 1},
        {"country": "Canada",    "n": 1},
    ]

    gen = MagicMock()
    gen.invoke.return_value = diagnostic_sql_returned

    with (
        patch("agent.nodes._sql_generator", return_value=gen),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
        patch("agent.nodes.run_query", return_value=diagnostic_rows),
    ):
        out = diagnose_empty({
            "question": "revenue for stores in the US over time",
            "data_question": "Monthly revenue for stores located in the United States",
            "sql": "SELECT ... WHERE co.country = 'United States' ...",
            "sql_error": None,
            "rows": [],
            "retries": 0,
        })

    assert out["diagnostic_rows"] == diagnostic_rows
    assert out["diagnostic_sql"]
    assert "country" in out["diagnostic_sql"]
    gen.invoke.assert_called_once()


def test_diagnostic_passes_failed_sql_into_prompt():
    """The LLM must see the original SQL to know what filter to drop."""
    failing_sql = "SELECT SUM(amount) FROM payment WHERE country = 'United States'"
    gen = MagicMock()
    gen.invoke.return_value = SQLGeneration(
        reasoning="(test)", tables_used=["country"],
        sql="SELECT country, COUNT(*) FROM country GROUP BY country LIMIT 20",
    )
    with (
        patch("agent.nodes._sql_generator", return_value=gen),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
        patch("agent.nodes.run_query", return_value=[]),
    ):
        diagnose_empty({
            "question": "x", "data_question": "x",
            "sql": failing_sql, "sql_error": None,
            "rows": [], "retries": 0,
        })
    msgs = gen.invoke.call_args.args[0]
    user_content = next(m for m in msgs if m["role"] == "user")["content"]
    assert failing_sql in user_content


# ---------- node-level: graceful failures ----------

def test_llm_error_returns_no_diagnostic():
    """LLM hiccup must not crash — degrade silently to 'no diagnostic'."""
    gen = MagicMock()
    gen.invoke.side_effect = RuntimeError("Anthropic API timeout")
    with (
        patch("agent.nodes._sql_generator", return_value=gen),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value=""),
    ):
        out = diagnose_empty({
            "question": "x", "data_question": "x",
            "sql": "SELECT * FROM x WHERE col = 'y'", "sql_error": None,
            "rows": [], "retries": 0,
        })
    assert out == {"diagnostic_sql": None, "diagnostic_rows": None}


def test_guard_rejection_returns_no_diagnostic():
    """If the LLM emits something the validator rejects, fall through."""
    gen = MagicMock()
    gen.invoke.return_value = SQLGeneration(
        reasoning="bad", tables_used=["x"], sql="DELETE FROM customer",
    )
    with (
        patch("agent.nodes._sql_generator", return_value=gen),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value=""),
    ):
        out = diagnose_empty({
            "question": "x", "data_question": "x",
            "sql": "SELECT 1 WHERE col = 'y'", "sql_error": None,
            "rows": [], "retries": 0,
        })
    assert out == {"diagnostic_sql": None, "diagnostic_rows": None}


def test_executor_error_on_diagnostic_returns_no_diagnostic():
    """If the diagnostic SQL itself errors at the DB, fall through cleanly."""
    gen = MagicMock()
    gen.invoke.return_value = SQLGeneration(
        reasoning="x", tables_used=["country"],
        sql="SELECT country, COUNT(*) FROM country GROUP BY country LIMIT 20",
    )
    with (
        patch("agent.nodes._sql_generator", return_value=gen),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value=""),
        patch("agent.nodes.run_query", side_effect=RuntimeError("connection lost")),
    ):
        out = diagnose_empty({
            "question": "x", "data_question": "x",
            "sql": "SELECT 1 WHERE col = 'y'", "sql_error": None,
            "rows": [], "retries": 0,
        })
    assert out == {"diagnostic_sql": None, "diagnostic_rows": None}


# ---------- end-to-end through the graph ----------

@pytest.mark.asyncio
async def test_empty_result_routes_through_diagnose_empty_to_summarize():
    """Full graph path: data question → SQL runs → 0 rows →
    diagnose_empty fires → summarize sees diagnostic_rows in state."""
    front_llm = MagicMock()
    front_llm.invoke.return_value = FrontAgentDecision(
        intent="data",
        data_question="Monthly revenue for stores in the United States, 2022",
    )

    sql_main = SQLGeneration(
        reasoning="(test)", tables_used=["payment", "store", "country"],
        sql="SELECT 1 FROM country WHERE country = 'United States'",
    )
    sql_diag = SQLGeneration(
        reasoning="diagnostic", tables_used=["country"],
        sql="SELECT country, COUNT(*) AS n FROM country GROUP BY country LIMIT 20",
    )
    sql_gen = MagicMock()
    # Two calls: main SQL + diagnostic SQL
    sql_gen.invoke.side_effect = [sql_main, sql_diag]

    chart_picker = MagicMock()
    chart_picker.invoke.return_value = ChartSpec(kind="none", reasoning="empty result")

    summary_llm = MagicMock()
    summary_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content="No US stores in the data. Stores are in Australia, Canada."
    ))

    # First run_query (main): empty. Second run_query (diagnostic): has data.
    diagnostic_rows = [
        {"country": "Australia", "n": 1},
        {"country": "Canada",    "n": 1},
    ]
    run_query_calls = [[], diagnostic_rows]

    def fake_run_query(sql):
        return run_query_calls.pop(0)

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._sql_generator", return_value=sql_gen),
        patch("agent.nodes._summarizer", return_value=summary_llm),
        patch("agent.nodes._chart_picker", return_value=chart_picker),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<vocab>"),
        patch("agent.nodes.run_query", side_effect=fake_run_query),
    ):
        config = {"configurable": {"thread_id": "diag-test"}}
        final_state = await app_graph.ainvoke(
            turn_input("revenue for stores in the US over time",
                       HumanMessage(content="revenue for stores in the US over time")),
            config=config,
        )

    # Main query ran, returned empty.
    assert final_state["rows"] == []
    # Diagnostic ran and surfaced the actual countries with stores.
    assert final_state["diagnostic_rows"] == diagnostic_rows
    assert final_state["diagnostic_sql"]
    # SQL gen was called twice: main + diagnostic
    assert sql_gen.invoke.call_count == 2
    # Summarize received the diagnostic in its prompt
    summarize_msgs = summary_llm.ainvoke.call_args.args[0]
    summarize_user = next(m for m in summarize_msgs if m["role"] == "user")["content"]
    assert "Diagnostic results" in summarize_user
    assert "Australia" in summarize_user
    assert "Canada" in summarize_user


@pytest.mark.asyncio
async def test_non_empty_result_skips_diagnose_empty():
    """Sanity: when the main query returns rows, diagnose_empty must NOT fire."""
    front_llm = MagicMock()
    front_llm.invoke.return_value = FrontAgentDecision(
        intent="data", data_question="any",
    )

    sql_gen = MagicMock()
    sql_gen.invoke.return_value = SQLGeneration(
        reasoning="x", tables_used=["x"], sql="SELECT 1 AS n",
    )

    chart_picker = MagicMock()
    chart_picker.invoke.return_value = ChartSpec(kind="none")

    summary_llm = MagicMock()
    summary_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._sql_generator", return_value=sql_gen),
        patch("agent.nodes._summarizer", return_value=summary_llm),
        patch("agent.nodes._chart_picker", return_value=chart_picker),
        patch("agent.nodes.pagila_schema_string", return_value="<schema>"),
        patch("agent.nodes.vocabulary_string", return_value=""),
        patch("agent.nodes.run_query", return_value=[{"n": 1}, {"n": 2}]),
    ):
        config = {"configurable": {"thread_id": "diag-skip-test"}}
        final_state = await app_graph.ainvoke(
            turn_input("anything", HumanMessage(content="anything")),
            config=config,
        )

    assert final_state["rows"] == [{"n": 1}, {"n": 2}]
    assert final_state.get("diagnostic_rows") is None
    # Main SQL gen called exactly once — no diagnostic call
    assert sql_gen.invoke.call_count == 1
