import json
from functools import cache

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.exc import DBAPIError, OperationalError

from agent.db import pagila_schema_string, run_query
from agent.prompts import (
    DATASET_NOTES,
    FRONT_AGENT_SYSTEM,
    FRONT_AGENT_USER,
    SQL_GENERATION_SYSTEM,
    SQL_GENERATION_USER,
    SUMMARIZE_SYSTEM,
    SUMMARIZE_USER,
)
from agent.schemas import FrontAgentDecision, SQLGeneration
from agent.state import AgentState

MAX_RETRIES = 2
SUMMARIZE_ROW_BUDGET = 50
HISTORY_TURN_LIMIT = 12  # last N messages shown to front agent


@cache
def _front_agent_llm():
    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    return llm.with_structured_output(FrontAgentDecision)


@cache
def _sql_generator():
    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    return llm.with_structured_output(SQLGeneration)


@cache
def _summarizer():
    # Plain LLM (no structured output) so token streaming has clean text chunks
    return ChatAnthropic(model="claude-sonnet-4-5", temperature=0)


def _format_history(messages: list[BaseMessage], current_question: str) -> str:
    # Drop the just-arrived HumanMessage (it's shown separately in the prompt)
    # and trim to the last N for prompt cost control.
    prior = list(messages)
    if prior and isinstance(prior[-1], HumanMessage) and prior[-1].content == current_question:
        prior = prior[:-1]
    if not prior:
        return "(no prior turns)"
    prior = prior[-HISTORY_TURN_LIMIT:]
    lines = []
    for m in prior:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _rows_block(rows: list[dict] | None, sql_error: str | None) -> str:
    if sql_error and not rows:
        return f"The query failed with this error:\n{sql_error}\n\nNo rows are available."
    if not rows:
        return "The query returned 0 rows."
    head = rows[:SUMMARIZE_ROW_BUDGET]
    truncated = len(rows) > len(head)
    note = f"\n\n(Showing first {len(head)} of {len(rows)} rows.)" if truncated else ""
    payload = json.dumps(head, default=str, indent=2)
    return f"Rows ({len(rows)} total):\n```json\n{payload}\n```{note}"


def front_agent(state: AgentState) -> dict:
    """Conversational front-end. Either replies directly or hands a clean question to SQL gen."""
    system = FRONT_AGENT_SYSTEM.format(dataset_notes=DATASET_NOTES)
    user = FRONT_AGENT_USER.format(
        conversation=_format_history(state.get("messages") or [], state["question"]),
        question=state["question"],
    )
    decision: FrontAgentDecision = _front_agent_llm().invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )

    if decision.intent == "data":
        return {"intent": "data", "data_question": decision.data_question}

    return {
        "intent": "respond",
        "summary": decision.response_text,
        "messages": [AIMessage(content=decision.response_text)],
    }


def generate_sql(state: AgentState) -> dict:
    question = state.get("data_question") or state["question"]
    system = SQL_GENERATION_SYSTEM.format(
        dataset_notes=DATASET_NOTES,
        schema=pagila_schema_string(),
    )
    user = SQL_GENERATION_USER.format(question=question)

    result: SQLGeneration = _sql_generator().invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    return {"sql": result.sql, "sql_error": None}


def validate_sql(state: AgentState) -> dict:
    # Stub: always passes for now
    return {"sql": state["sql"], "sql_error": None}


def execute_sql(state: AgentState) -> dict:
    """Execute the validated SQL. Route errors back for retry with specifics."""
    try:
        rows = run_query(state["sql"])
        return {"rows": rows, "sql_error": None}

    except OperationalError as e:
        msg = str(e.orig) if e.orig else str(e)
        if "statement timeout" in msg.lower():
            return {
                "sql_error": (
                    "Query exceeded 10 second timeout. "
                    "Rewrite to be more selective — add WHERE clauses, "
                    "reduce joins, or aggregate earlier."
                ),
                "retries": state["retries"] + 1,
            }
        return {
            "sql_error": f"Database connection error: {msg}",
            "retries": state["retries"] + 1,
        }

    except DBAPIError as e:
        msg = str(e.orig) if e.orig else str(e)
        return {
            "sql_error": msg,
            "retries": state["retries"] + 1,
        }

    except Exception as e:
        return {
            "sql_error": f"Unexpected error: {type(e).__name__}: {e}",
            "retries": state["retries"] + 1,
        }


def summarize(state: AgentState) -> dict:
    question = state.get("data_question") or state["question"]
    system = SUMMARIZE_SYSTEM.format(dataset_notes=DATASET_NOTES)
    user = SUMMARIZE_USER.format(
        question=question,
        sql=state.get("sql") or "(no SQL was generated)",
        rows_block=_rows_block(state.get("rows"), state.get("sql_error")),
    )
    msg = _summarizer().invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    summary_text = msg.content if isinstance(msg.content, str) else "".join(
        b.get("text", "") for b in msg.content if isinstance(b, dict) and b.get("type") == "text"
    )
    return {
        "summary": summary_text,
        "messages": [AIMessage(content=summary_text)],
    }
