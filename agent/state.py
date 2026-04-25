from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class ChartSpec(BaseModel):
    kind: Literal["bar", "line", "pie", "table", "none"]
    x: str | None = None       # column for category / time axis
    y: str | None = None       # column for the numeric measure
    group: str | None = None   # optional 3rd column for color grouping (bar/line only)
    title: str | None = None
    reasoning: str | None = None

    # Layout knobs — expose just enough plotly express args to handle common
    # user follow-ups like "stack the bars", "make it horizontal", "facet per genre",
    # "sort by count". All optional; bar/line only (pie ignores them).
    barmode: Literal["group", "stack", "relative", "overlay"] | None = None
    orientation: Literal["v", "h"] | None = None
    facet_col: str | None = None       # subplot dimension (column from result)
    sort_by: str | None = None         # column to sort x-axis by
    sort_desc: bool | None = None      # True for descending


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str                    # current turn's raw user input
    intent: Literal["data", "respond", "rechart"] | None
    data_question: str | None        # front agent's refined, self-contained restatement for SQL gen
    sql: str | None
    sql_error: str | None
    rows: list[dict] | None
    summary: str | None
    chart: ChartSpec | None
    chart_kind_override: Literal["bar", "line", "pie", "table"] | None
    retries: int
    # Diagnostic surfaced to summarize when the main query returned 0 rows.
    # diagnose_empty fills these so the answer can suggest "no rows for X,
    # but here's what IS available" rather than a flat "no rows".
    diagnostic_sql: str | None
    diagnostic_rows: list[dict] | None


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
    }
