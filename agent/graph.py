# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from agent.state import AgentState

# Register our custom Pydantic types with the checkpoint serializer.
# Without this, LangGraph 1.x logs a "Deserializing unregistered type" warning
# every time state containing a ChartSpec is rehydrated, and will hard-block
# in a future version.
_checkpoint_serde = JsonPlusSerializer(
    allowed_msgpack_modules=[
        ("agent.state", "ChartCode"),
        ("agent.schemas", "ReportSection"),
    ],
)
from agent.nodes import (
    front_agent, retrieve_context, generate_sql, validate_sql,
    execute_sql, diagnose_empty, summarize, generate_chart,
    generate_response,
    plan_report, warmup_sql_cache, sub_query, aggregate_report,
    MAX_RETRIES,
)


def after_front(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "data":
        # Data path goes through retrieve_context first so the SQL generator
        # sees question-specific glossary + few-shot examples baked into
        # its user message.
        return "retrieve_context"
    if intent == "report":
        return "plan_report"
    # Deterministic safety refusal short-circuits with summary already set
    # in front_agent — skip generate_response, go straight to END.
    if state.get("summary"):
        return "END"
    # respond / rechart → stream a real reply via generate_response
    return "generate_response"


def fan_out_sections(state: AgentState):
    """After plan_report runs, fan out to one parallel sub_query per section.

    Returns a list of Send objects — LangGraph dispatches each in parallel
    with its own `current_section` payload. After all branches complete,
    the regular `sub_query → aggregate_report` edge fires once.
    """
    outline = state.get("report_outline") or []
    return [
        Send("sub_query", {"current_section": section, "question": state["question"]})
        for section in outline
    ]


def after_generate_response(state: AgentState) -> str:
    if state["intent"] == "rechart":
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
    g.add_node("generate_response", generate_response)
    g.add_node("retrieve_context", retrieve_context)
    g.add_node("generate_sql", generate_sql)
    g.add_node("validate_sql", validate_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_node("diagnose_empty", diagnose_empty)
    g.add_node("summarize", summarize)
    g.add_node("generate_chart", generate_chart)
    # Report-mode nodes
    g.add_node("plan_report", plan_report)
    g.add_node("warmup_sql_cache", warmup_sql_cache)
    g.add_node("sub_query", sub_query)
    g.add_node("aggregate_report", aggregate_report)

    g.set_entry_point("front_agent")
    g.add_conditional_edges("front_agent", after_front, {
        "retrieve_context":  "retrieve_context",
        "generate_response": "generate_response",
        "plan_report":       "plan_report",
        "END":               END,
    })
    # Data path picks up retrieved glossary + examples, then proceeds
    # to SQL generation. The retry loop (validate → generate_sql) reuses
    # the already-fetched context — no need to re-retrieve on retry.
    g.add_edge("retrieve_context", "generate_sql")
    # Report path: plan_report → warmup_sql_cache → fan-out via Send →
    # parallel sub_query → aggregate_report. The warmup serializes a single
    # cache write so the parallel sub_queries that follow get cache READS
    # instead of all paying the write premium simultaneously (no-op if the
    # cache is already warm from a previous report within ~5 minutes).
    # LangGraph waits for all parallel sub_query branches to complete (the
    # report_sections reducer accumulates their results) before firing
    # aggregate_report.
    g.add_edge("plan_report", "warmup_sql_cache")
    g.add_conditional_edges("warmup_sql_cache", fan_out_sections, ["sub_query"])
    g.add_edge("sub_query", "aggregate_report")
    g.add_edge("aggregate_report", END)
    g.add_conditional_edges("generate_response", after_generate_response, {
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

    return g.compile(checkpointer=MemorySaver(serde=_checkpoint_serde))


# Module-level singleton — Chainlit imports this
app_graph = build_graph()
