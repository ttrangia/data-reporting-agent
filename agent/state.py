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
    question: str                    # current turn's raw user input
    intent: Literal["data", "respond"] | None
    data_question: str | None        # front agent's refined, self-contained restatement for SQL gen
    sql: str | None
    sql_error: str | None
    rows: list[dict] | None
    summary: str | None
    chart: ChartSpec | None
    retries: int


def turn_input(question: str, human_message: BaseMessage) -> dict:
    """Per-turn delta to merge into checkpointed state.

    `messages` is appended via add_messages reducer so history accumulates.
    Other fields are reset so prior-turn values don't leak.
    """
    return {
        "messages": [human_message],
        "question": question,
        "intent": None,
        "data_question": None,
        "sql": None,
        "sql_error": None,
        "rows": None,
        "summary": None,
        "chart": None,
        "retries": 0,
    }
