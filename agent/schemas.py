from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FrontAgentDecision(BaseModel):
    intent: Literal["data", "respond"] = Field(
        description=(
            "'data' if the user is asking a clear, answerable data question; "
            "'respond' for chat, clarification, or out-of-scope replies."
        )
    )
    response_text: str | None = Field(
        default=None,
        description="When intent='respond', the message to send to the user. Null when intent='data'.",
    )
    data_question: str | None = Field(
        default=None,
        description=(
            "When intent='data', a self-contained restatement of what to query — "
            "pronouns resolved, time scope explicit. Null when intent='respond'."
        ),
    )

    @model_validator(mode="after")
    def _check_field_for_intent(self) -> "FrontAgentDecision":
        if self.intent == "data" and not self.data_question:
            raise ValueError("intent='data' requires a non-empty data_question")
        if self.intent == "respond" and not self.response_text:
            raise ValueError("intent='respond' requires a non-empty response_text")
        return self


class ReportOutput(BaseModel):
    summary: str = Field(
        description=(
            "A direct natural-language answer to the user's question, grounded in the rows. "
            "2-4 sentences. Cite specific numbers from the data. "
            "Do NOT invent values that aren't present in the rows."
        )
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Optional bulletable highlights from the data — e.g. notable outliers, "
            "totals, or comparisons. Empty list if the result is a single number or trivially small."
        ),
    )


class SQLGeneration(BaseModel):
    reasoning: str = Field(
        description=(
            "One or two sentences explaining which tables you chose, "
            "how they join, and any filter/aggregation logic."
        )
    )
    tables_used: list[str] = Field(
        description="Tables referenced in the query, unqualified (e.g. 'rental', 'customer')."
    )
    sql: str = Field(
        description=(
            "A single Postgres SELECT statement answering the question. "
            "No DDL, no DML, no semicolons separating multiple statements."
        )
    )
