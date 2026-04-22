from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class ChartSpec(BaseModel):
    kind: Literal["bar", "line", "pie", "table"]
    x: str | None = None
    y: str | None = None


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    intent: Literal["data", "chat", "unclear"] | None
    sql: str | None
    sql_error: str | None
    rows: list[dict] | None
    summary: str | None
    chart: ChartSpec | None
    retries: int


def initial_state(question: str) -> AgentState:
    """Seed state for a fresh turn."""
    return {
        "messages": [],
        "question": question,
        "intent": None,
        "sql": None,
        "sql_error": None,
        "rows": None,
        "summary": None,
        "chart": None,
        "retries": 0,
    }