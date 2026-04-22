# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    classify_intent, generate_sql, validate_sql,
    execute_sql, summarize, handle_chat,
    MAX_RETRIES,
)


def after_intent(state: AgentState) -> str:
    return "generate_sql" if state["intent"] == "data" else "handle_chat"


def after_validate(state: AgentState) -> str:
    if state["sql_error"]:
        if state["retries"] >= MAX_RETRIES:
            return "summarize"
        return "generate_sql"
    return "execute_sql"


def after_execute(state: AgentState) -> str:
    if state["sql_error"]:
        if state["retries"] >= MAX_RETRIES:
            return "summarize"
        return "generate_sql"
    return "summarize"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("generate_sql", generate_sql)
    g.add_node("validate_sql", validate_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_node("summarize", summarize)
    g.add_node("handle_chat", handle_chat)

    g.set_entry_point("classify_intent")
    g.add_conditional_edges("classify_intent", after_intent, {
        "generate_sql": "generate_sql",
        "handle_chat": "handle_chat",
    })
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges("validate_sql", after_validate, {
        "generate_sql": "generate_sql",
        "execute_sql": "execute_sql",
        "summarize": "summarize",
    })
    g.add_conditional_edges("execute_sql", after_execute, {
        "generate_sql": "generate_sql",
        "summarize": "summarize",
    })
    g.add_edge("summarize", END)
    g.add_edge("handle_chat", END)

    return g.compile(checkpointer=MemorySaver())


# Module-level singleton — Chainlit imports this
app_graph = build_graph()