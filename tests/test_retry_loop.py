"""Tests for the generate→validate→retry loop.

Mocks every LLM and the DB so the graph runs deterministically. The point is
to verify graph topology (does it loop back?) and prompt content (does the
retry hint actually reach the LLM on attempt 2?), not real model behavior.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.graph import app_graph
from agent.nodes import MAX_RETRIES
from agent.schemas import FrontAgentDecision, SQLGeneration
from agent.state import turn_input


def _front_decision(question: str) -> FrontAgentDecision:
    return FrontAgentDecision(intent="data", data_question=question)


def _sql_gen(sql: str) -> SQLGeneration:
    return SQLGeneration(reasoning="(test)", tables_used=["customer"], sql=sql)


@pytest.mark.asyncio
async def test_retry_recovers_after_validator_rejection():
    """First-attempt SQL violates the guard. Graph should loop back to
    generate_sql, the retry prompt should include the prior SQL + guard
    error, and the second-attempt good SQL should be executed and summarized.
    """
    front_llm = MagicMock()
    front_llm.invoke.return_value = _front_decision("show the first customer")

    sql_llm = MagicMock()
    sql_llm.invoke.side_effect = [
        _sql_gen("DELETE FROM customer"),                          # rejected by guard
        _sql_gen("SELECT customer_id FROM customer LIMIT 1"),       # good
    ]

    summary_llm = MagicMock()
    summary_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Customer 1."))

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._sql_generator", return_value=sql_llm),
        patch("agent.nodes._summarizer", return_value=summary_llm),
        patch("agent.nodes.pagila_schema_string", return_value="<fake schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<fake vocab>"),
        patch("agent.nodes.run_query", return_value=[{"customer_id": 1}]),
    ):
        final_state = await app_graph.ainvoke(
            turn_input("show the first customer", HumanMessage(content="show the first customer")),
            config={"configurable": {"thread_id": "retry-recover"}},
        )

    # Retry counter incremented once (validator failed once)
    assert final_state["retries"] == 1
    # Final SQL is the good one, canonicalized by the guard
    assert "SELECT" in final_state["sql"].upper()
    assert "DELETE" not in final_state["sql"].upper()
    # Rows fetched and summary produced
    assert final_state["rows"] == [{"customer_id": 1}]
    assert final_state["summary"] == "Customer 1."
    # SQL generator was called exactly twice
    assert sql_llm.invoke.call_count == 2
    # The second call's user message contains the retry hint with the prior SQL + error
    second_call_msgs = sql_llm.invoke.call_args_list[1].args[0]
    user_msg = next(m for m in second_call_msgs if m["role"] == "user")
    assert "Retry context" in user_msg["content"]
    assert "DELETE FROM customer" in user_msg["content"]
    assert "SQL guard rejected" in user_msg["content"]


@pytest.mark.asyncio
async def test_retry_recovers_after_executor_error():
    """First-attempt SQL passes the guard but Postgres rejects it (e.g., bad
    column). Graph should loop back, retry with the executor's error in the
    prompt, and execute the corrected SQL on attempt 2.
    """
    from sqlalchemy.exc import DBAPIError

    front_llm = MagicMock()
    front_llm.invoke.return_value = _front_decision("first customer email")

    sql_llm = MagicMock()
    sql_llm.invoke.side_effect = [
        _sql_gen("SELECT email_address FROM customer LIMIT 1"),    # column doesn't exist
        _sql_gen("SELECT email FROM customer LIMIT 1"),            # corrected
    ]

    summary_llm = MagicMock()
    summary_llm.ainvoke = AsyncMock(return_value=AIMessage(content="x@y.com."))

    pg_error = DBAPIError.instance(
        statement="SELECT email_address FROM customer LIMIT 1",
        params=None,
        orig=Exception('column "email_address" does not exist'),
        dbapi_base_err=Exception,
    )

    def fake_run_query(sql: str):
        if "email_address" in sql:
            raise pg_error
        return [{"email": "x@y.com"}]

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._sql_generator", return_value=sql_llm),
        patch("agent.nodes._summarizer", return_value=summary_llm),
        patch("agent.nodes.pagila_schema_string", return_value="<fake schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<fake vocab>"),
        patch("agent.nodes.run_query", side_effect=fake_run_query),
    ):
        final_state = await app_graph.ainvoke(
            turn_input("first customer email", HumanMessage(content="first customer email")),
            config={"configurable": {"thread_id": "retry-exec"}},
        )

    assert final_state["retries"] == 1
    assert "email_address" not in final_state["sql"]
    assert final_state["rows"] == [{"email": "x@y.com"}]
    # Retry hint surfaced the executor's column-not-found error
    second_call_msgs = sql_llm.invoke.call_args_list[1].args[0]
    user_msg = next(m for m in second_call_msgs if m["role"] == "user")
    assert "email_address" in user_msg["content"]
    assert "does not exist" in user_msg["content"]


@pytest.mark.asyncio
async def test_retry_exhaustion_routes_to_summarize_with_error():
    """Persistent failure: every attempt produces guard-rejected SQL. After
    retries hit MAX_RETRIES, the graph should give up and route to summarize
    with the error in state — graceful degradation, not an infinite loop.
    """
    front_llm = MagicMock()
    front_llm.invoke.return_value = _front_decision("anything")

    sql_llm = MagicMock()
    sql_llm.invoke.return_value = _sql_gen("DELETE FROM customer")

    summary_llm = MagicMock()
    summary_llm.ainvoke = AsyncMock(return_value=AIMessage(content="I couldn't answer."))

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._sql_generator", return_value=sql_llm),
        patch("agent.nodes._summarizer", return_value=summary_llm),
        patch("agent.nodes.pagila_schema_string", return_value="<fake schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<fake vocab>"),
        patch("agent.nodes.run_query", return_value=[]),
    ):
        final_state = await app_graph.ainvoke(
            turn_input("foo", HumanMessage(content="foo")),
            config={"configurable": {"thread_id": "retry-exhaust"}},
        )

    assert final_state["retries"] == MAX_RETRIES
    assert final_state["sql_error"]                       # error preserved for summarize
    assert "guard rejected" in final_state["sql_error"].lower()
    assert final_state["summary"] == "I couldn't answer." # graceful degradation
    # Graph stopped retrying — exactly MAX_RETRIES attempts (each fails validation)
    assert sql_llm.invoke.call_count == MAX_RETRIES


@pytest.mark.asyncio
async def test_first_attempt_prompt_has_no_retry_hint():
    """Sanity check the converse: when there's no prior error, the generation
    prompt should NOT include retry context — otherwise the LLM gets confused.
    """
    front_llm = MagicMock()
    front_llm.invoke.return_value = _front_decision("first customer")

    sql_llm = MagicMock()
    sql_llm.invoke.return_value = _sql_gen("SELECT customer_id FROM customer LIMIT 1")

    summary_llm = MagicMock()
    summary_llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

    with (
        patch("agent.nodes._front_agent_llm", return_value=front_llm),
        patch("agent.nodes._sql_generator", return_value=sql_llm),
        patch("agent.nodes._summarizer", return_value=summary_llm),
        patch("agent.nodes.pagila_schema_string", return_value="<fake schema>"),
        patch("agent.nodes.vocabulary_string", return_value="<fake vocab>"),
        patch("agent.nodes.run_query", return_value=[{"customer_id": 1}]),
    ):
        await app_graph.ainvoke(
            turn_input("first customer", HumanMessage(content="first customer")),
            config={"configurable": {"thread_id": "no-retry"}},
        )

    assert sql_llm.invoke.call_count == 1
    msgs = sql_llm.invoke.call_args.args[0]
    user_msg = next(m for m in msgs if m["role"] == "user")
    assert "Retry context" not in user_msg["content"]
    assert user_msg["content"].startswith("Question:")
