# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    front_agent, generate_sql, validate_sql,
    execute_sql, diagnose_empty, summarize, generate_chart,
    MAX_RETRIES,
)


def after_front(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "data":
        return "generate_sql"
    if intent == "rechart":
        return "generate_chart"
    return "END"  # respond


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
    # Successful execute but 0 rows → run diagnostic before summarizing
    if not state.get("rows"):
        return "diagnose_empty"
    return "summarize"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("front_agent", front_agent)
    g.add_node("generate_sql", generate_sql)
    g.add_node("validate_sql", validate_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_node("diagnose_empty", diagnose_empty)
    g.add_node("summarize", summarize)
    g.add_node("generate_chart", generate_chart)

    g.set_entry_point("front_agent")
    g.add_conditional_edges("front_agent", after_front, {
        "generate_sql":   "generate_sql",
        "generate_chart": "generate_chart",
        "END":            END,
    })
    g.add_edge("generate_sql", "validate_sql")
    g.add_conditional_edges("validate_sql", after_validate, {
        "generate_sql": "generate_sql",
        "execute_sql": "execute_sql",
        "summarize": "summarize",
    })
    g.add_conditional_edges("execute_sql", after_execute, {
        "generate_sql": "generate_sql",
        "diagnose_empty": "diagnose_empty",
        "summarize": "summarize",
    })
    g.add_edge("diagnose_empty", "summarize")
    g.add_edge("summarize", "generate_chart")
    g.add_edge("generate_chart", END)

    return g.compile(checkpointer=MemorySaver())


# Module-level singleton — Chainlit imports this
app_graph = build_graph()
