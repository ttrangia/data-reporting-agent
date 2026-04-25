from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agent.state import ChartCode


class FrontAgentDecision(BaseModel):
    """Routing-only decision from the classify-intent step.

    The reply text for respond/rechart paths is NOT generated here — a
    downstream `generate_response` node produces it via plain-text streaming.
    Keeping this schema purely structural lets the front-agent LLM stay
    cheap/fast and lets the response stream cleanly token-by-token.
    """
    intent: Literal["data", "respond", "rechart", "report"] = Field(
        description=(
            "'data' if the user is asking a clear, answerable single-metric data question; "
            "'rechart' if the user wants to re-render the previous query's data "
            "as a different chart kind (no new SQL); "
            "'report' for broad asks that warrant a multi-section structured report "
            "with multiple sub-queries (e.g. 'quarterly review', 'executive summary'); "
            "'respond' for chat, clarification, refusals, or out-of-scope replies."
        )
    )
    data_question: str | None = Field(
        default=None,
        description=(
            "When intent='data', a self-contained restatement of what to query — "
            "pronouns resolved, time scope explicit. Null otherwise."
        ),
    )
    chart_kind_override: Literal["bar", "line", "pie", "table"] | None = Field(
        default=None,
        description=(
            "When intent='rechart', the chart kind the user asked for. "
            "Use 'table' if the user wants the rendering as a table (no chart). "
            "Null otherwise."
        ),
    )

    @model_validator(mode="after")
    def _check_field_for_intent(self) -> "FrontAgentDecision":
        if self.intent == "data" and not self.data_question:
            raise ValueError("intent='data' requires a non-empty data_question")
        return self


class ReportSection(BaseModel):
    """One section of a multi-part report.

    Created by the planner with the first 4 fields populated; the rest are
    filled in by sub_query as it runs the SQL pipeline for this section.
    Lives in state.report_sections (an accumulator with a custom reducer)
    so parallel sub_query branches can each append their completed section.
    """
    title: str = Field(description="3-8 word section heading.")
    sub_question: str = Field(
        description=(
            "Self-contained data question for the SQL generator — must include "
            "explicit time scope and any disambiguation. Don't say 'this quarter', "
            "say 'August 2022'."
        )
    )
    chart_hint: Literal["bar", "line", "pie", "table", "none"] | None = Field(
        default=None,
        description=(
            "Best guess at the visualization for this section's results. "
            "'none' for headline metrics or text-shaped answers."
        ),
    )
    rationale: str | None = Field(
        default=None,
        description="One sentence on why this section belongs in the report.",
    )

    # Filled in by sub_query — None until the section completes
    sql: str | None = None
    row_count: int | None = None
    rows_preview: list[dict] | None = None  # capped at ~10 rows for aggregator context
    rows_for_chart: list[dict] | None = None  # larger sample (~500) used by chart sandbox
    summary: str | None = None
    chart: ChartCode | None = None
    error: str | None = None  # set if any step failed; section gracefully degraded


class ReportPlan(BaseModel):
    """Output of plan_report — decomposes a broad question into ordered sections."""
    sections: list[ReportSection] = Field(min_length=2, max_length=7)
    rationale: str = Field(
        description=(
            "One sentence: what overall question this report answers and "
            "how the sections fit together."
        )
    )


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
