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
    plan_report, warmup_sql_cache, dispatch_sections, sub_query,
    aggregate_report,
    MAX_RETRIES,
)


def after_front(state: AgentState):
    """Branch from front_agent.

    Report intent fans out to TWO parallel nodes (plan_report + warmup_sql_
    cache) by returning a list — LangGraph dispatches both concurrently and
    they reconverge at dispatch_sections. Without this parallel split,
    plan_report's ~3-15s would run sequentially before the 2-3s warmup,
    adding the warmup time to the critical path on every report turn.
    """
    intent = state["intent"]
    if intent == "data":
        # Data path goes through retrieve_context first so the SQL generator
        # sees question-specific glossary + few-shot examples baked into
        # its user message.
        return "retrieve_context"
    if intent == "report":
        return ["plan_report", "warmup_sql_cache"]
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


def after_validate(state: AgentState):
    """Route after sql_guard runs.

    Failure path is sequential (summarize alone) because there are no rows
    to chart — generate_chart short-circuits to None anyway, but skipping
    it removes an empty step from the UI.
    """
    if state["sql_error"]:
        if state["retries"] >= MAX_RETRIES:
            return "summarize"
        return "generate_sql"
    return "execute_sql"


def after_execute(state: AgentState):
    """Route after Postgres returns.

    Success path with rows fans out to BOTH summarize and generate_chart in
    parallel — they're independent (both read state.rows, neither needs
    the other's output) and serializing them adds the chart picker's
    4-8s of latency to the critical path on every data turn. Both nodes
    flow to END; their state updates merge naturally.
    """
    if state["sql_error"]:
        if state["retries"] >= MAX_RETRIES:
            return "summarize"
        return "generate_sql"
    # Successful execute but 0 rows → run diagnostic before summarizing.
    # diagnose_empty also fans out to [summarize, generate_chart] downstream.
    if not state.get("rows"):
        return "diagnose_empty"
    return ["summarize", "generate_chart"]


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
    g.add_node("dispatch_sections", dispatch_sections)
    g.add_node("sub_query", sub_query)
    g.add_node("aggregate_report", aggregate_report)

    g.set_entry_point("front_agent")
    g.add_conditional_edges("front_agent", after_front, {
        "retrieve_context":  "retrieve_context",
        "generate_response": "generate_response",
        "plan_report":       "plan_report",
        "warmup_sql_cache":  "warmup_sql_cache",  # parallel sibling of plan_report
        "END":               END,
    })
    # Data path picks up retrieved glossary + examples, then proceeds
    # to SQL generation. The retry loop (validate → generate_sql) reuses
    # the already-fetched context — no need to re-retrieve on retry.
    g.add_edge("retrieve_context", "generate_sql")
    # Report path: front_agent dispatches plan_report AND warmup_sql_cache
    # in parallel (see after_front). They both flow into dispatch_sections,
    # which is a no-op join that gates the fan-out — LangGraph waits for
    # both incoming edges before firing dispatch_sections, so we know
    # the outline is populated AND the prompt cache is warm before parallel
    # sub_queries dispatch. Saves 2-3s of serial warmup time per report.
    # After all sub_query branches complete, aggregate_report composes the
    # final markdown.
    g.add_edge("plan_report", "dispatch_sections")
    g.add_edge("warmup_sql_cache", "dispatch_sections")
    g.add_conditional_edges("dispatch_sections", fan_out_sections, ["sub_query"])
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
        "generate_chart": "generate_chart",  # parallel sibling of summarize
    })
    # diagnose_empty fans out to summarize + generate_chart too. The chart
    # node short-circuits to None when there are no rows, but running it
    # in parallel keeps the data-turn topology uniform. (Pure no-op cost
    # since generate_chart returns immediately on the empty-rows path.)
    g.add_edge("diagnose_empty", "summarize")
    g.add_edge("diagnose_empty", "generate_chart")
    # Both summarize and generate_chart are leaf nodes. LangGraph waits for
    # all in-flight branches to terminate before declaring the run complete,
    # so the final state always has both `summary` and `chart` populated
    # (or `chart=None` if the picker declined / there were no rows).
    g.add_edge("summarize", END)
    g.add_edge("generate_chart", END)

    return g.compile(checkpointer=MemorySaver(serde=_checkpoint_serde))


# Module-level singleton — Chainlit imports this
app_graph = build_graph()
