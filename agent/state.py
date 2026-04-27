from typing import Annotated, Any, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


def _merge_report_sections(prior: list | None, new: Any) -> list:
    """Reducer for report_sections.

    Receiving None resets to empty list — used by plan_report to clear
    the accumulator at the start of each report turn (the default
    `operator.add` reducer can't reset because `prior + [] == prior`).
    Receiving a list appends — used by parallel sub_query branches.
    """
    if new is None:
        return []
    if isinstance(new, list):
        return (prior or []) + new
    return prior or []


class ChartCode(BaseModel):
    """LLM-authored chart specification.

    Replaces the old declarative ChartSpec (kind/x/y/group/...) with executable
    Python code that the model writes against a `df: pd.DataFrame` and returns
    via a `fig` variable. The sandbox in agent.chart_sandbox enforces safety.

    Why code instead of a fixed schema? plotly.express's defaults don't always
    match the data shape (axis ordering, label formats, log scales, etc.).
    Letting the LLM write transform + plot code gives chart quality on par
    with what Claude Chat produces in its sandbox, without requiring us to
    enumerate every possible plotly knob in a schema.
    """
    reasoning: str | None = None      # one sentence: what this chart shows
    code: str | None = None           # Python that produces a `fig` variable
    title: str | None = None          # short caption — useful for traceability/UI


# Deprecated alias — old tests and external callers still reference ChartSpec.
# New code must use ChartCode. Schema is the same; just a different name.
ChartSpec = ChartCode


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str                    # current turn's raw user input
    intent: Literal["data", "respond", "rechart", "report"] | None
    data_question: str | None        # front agent's refined, self-contained restatement for SQL gen
    sql: str | None
    sql_error: str | None
    rows: list[dict] | None
    summary: str | None
    chart: ChartCode | None
    chart_kind_override: Literal["bar", "line", "pie", "table"] | None
    retries: int
    # Diagnostic surfaced to summarize when the main query returned 0 rows.
    # diagnose_empty fills these so the answer can suggest "no rows for X,
    # but here's what IS available" rather than a flat "no rows".
    diagnostic_sql: str | None
    diagnostic_rows: list[dict] | None
    # RAG-retrieved glossary + few-shot example block, baked into the SQL
    # generator's user prompt. None on respond/rechart paths (no SQL gen).
    # Kept in the user message (not system) to preserve cache-prefix hits.
    retrieved_context: str | None
    # Lightweight observability metadata for the retrieve_context node.
    # Lists of {id, similarity} so the Step body can show what was used
    # without dumping the full payload (which is in retrieved_context).
    retrieved_glossary: list[dict] | None
    retrieved_examples: list[dict] | None
    # Report-mode fields. report_outline is the planner's blueprint;
    # report_sections accumulates one entry per parallel sub_query (custom
    # reducer to support reset-via-None); report_text is the final assembled
    # markdown from aggregate_report. current_section is the per-Send payload
    # carrying which section the parallel sub_query is processing.
    report_outline: list[Any] | None             # list[ReportSection], avoid circular import
    report_plan_rationale: str | None
    report_sections: Annotated[list[Any], _merge_report_sections]
    report_text: str | None
    current_section: Any | None


def turn_input(question: str, human_message: BaseMessage) -> dict:
    """Per-turn delta to merge into checkpointed state.

    Reset (turn-scoped, must not leak across turns):
        question, intent, sql_error, summary, chart_kind_override, retries

    Preserved (cross-turn memory of the most recent meaningful query —
    enables follow-ups like 'replot as pie' to inherit the original
    descriptive caption rather than the literal rechart command):
        rows, sql, chart, data_question

    `messages` always appends via the add_messages reducer.
    """
    return {
        "messages": [human_message],
        "question": question,
        "intent": None,
        "sql_error": None,
        "summary": None,
        "chart_kind_override": None,
        "retries": 0,
        # Report fields reset every turn — they're scoped to the current report
        # and must not leak in from a prior turn.
        "report_outline": None,
        "report_plan_rationale": None,
        "report_sections": None,  # custom reducer interprets None as "reset to []"
        "report_text": None,
        "current_section": None,
        # RAG retrieval is per-question — clear from prior turn so a follow-up
        # gets fresh hits keyed off its own question text.
        "retrieved_context": None,
        "retrieved_glossary": None,
        "retrieved_examples": None,
    }
