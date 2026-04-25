import json
from functools import cache

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.exc import DBAPIError, OperationalError

from agent.db import pagila_schema_string, run_query
from agent.chart_directive import detect as detect_chart_directive
from agent.prompts import (
    CHART_FORCE_NOTE,
    CHART_SPEC_SYSTEM,
    CHART_SPEC_USER,
    DATASET_NOTES,
    FRONT_AGENT_SYSTEM,
    FRONT_AGENT_USER,
    SQL_GENERATION_RETRY_HINT,
    SQL_GENERATION_SYSTEM,
    SQL_GENERATION_USER,
    SUMMARIZE_SYSTEM,
    SUMMARIZE_USER,
)
from agent.safety import check_input
from agent.schemas import FrontAgentDecision, SQLGeneration
from agent.sql_guard import guard
from agent.state import AgentState, ChartSpec

MAX_RETRIES = 2
SUMMARIZE_ROW_BUDGET = 50
HISTORY_TURN_LIMIT = 12   # last N messages shown to front agent
CHART_ROW_BUDGET = 20     # rows shown to chart picker — enough to see shape, cheap to send


@cache
def _front_agent_llm():
    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    return llm.with_structured_output(FrontAgentDecision)


@cache
def _sql_generator():
    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    return llm.with_structured_output(SQLGeneration)


@cache
def _summarizer():
    # Plain LLM (no structured output) so token streaming has clean text chunks
    return ChatAnthropic(model="claude-sonnet-4-5", temperature=0)


@cache
def _chart_picker():
    llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
    return llm.with_structured_output(ChartSpec)


def _format_history(messages: list[BaseMessage], current_question: str) -> str:
    # Drop the just-arrived HumanMessage (it's shown separately in the prompt)
    # and trim to the last N for prompt cost control.
    prior = list(messages)
    if prior and isinstance(prior[-1], HumanMessage) and prior[-1].content == current_question:
        prior = prior[:-1]
    if not prior:
        return "(no prior turns)"
    prior = prior[-HISTORY_TURN_LIMIT:]
    lines = []
    for m in prior:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _rows_block(rows: list[dict] | None, sql_error: str | None) -> str:
    if sql_error and not rows:
        return f"The query failed with this error:\n{sql_error}\n\nNo rows are available."
    if not rows:
        return "The query returned 0 rows."
    head = rows[:SUMMARIZE_ROW_BUDGET]
    truncated = len(rows) > len(head)
    note = f"\n\n(Showing first {len(head)} of {len(rows)} rows.)" if truncated else ""
    payload = json.dumps(head, default=str, indent=2)
    return f"Rows ({len(rows)} total):\n```json\n{payload}\n```{note}"


def front_agent(state: AgentState) -> dict:
    """Conversational front-end. Either replies directly or hands a clean question to SQL gen.

    Deterministic safety gate runs first — obvious prompt-injection / bulk-PII /
    over-length inputs short-circuit to a refusal without spending an LLM call.
    The LLM-layer policy in FRONT_AGENT_SYSTEM handles nuanced cases that slip
    past the regex gate.
    """
    refusal = check_input(state["question"])
    if refusal is not None:
        return {
            "intent": "respond",
            "summary": refusal,
            "messages": [AIMessage(content=refusal)],
        }

    system = FRONT_AGENT_SYSTEM.format(dataset_notes=DATASET_NOTES)
    user = FRONT_AGENT_USER.format(
        conversation=_format_history(state.get("messages") or [], state["question"]),
        question=state["question"],
    )
    decision: FrontAgentDecision = _front_agent_llm().invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )

    if decision.intent == "data":
        return {"intent": "data", "data_question": decision.data_question}

    if decision.intent == "rechart":
        return {
            "intent": "rechart",
            "summary": decision.response_text,
            "chart_kind_override": decision.chart_kind_override,
            "messages": [AIMessage(content=decision.response_text)],
        }

    return {
        "intent": "respond",
        "summary": decision.response_text,
        "messages": [AIMessage(content=decision.response_text)],
    }


def generate_sql(state: AgentState) -> dict:
    question = state.get("data_question") or state["question"]

    # Retry path: include the prior SQL and the specific error so the LLM has
    # something concrete to fix instead of regenerating from scratch.
    if state.get("sql_error") and state.get("sql"):
        retry_context = SQL_GENERATION_RETRY_HINT.format(
            prior_sql=state["sql"],
            prior_error=state["sql_error"],
        )
    else:
        retry_context = ""

    system = SQL_GENERATION_SYSTEM.format(
        dataset_notes=DATASET_NOTES,
        schema=pagila_schema_string(),
    )
    user = SQL_GENERATION_USER.format(retry_context=retry_context, question=question)

    result: SQLGeneration = _sql_generator().invoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    # Clear sql_error so a passing validate_sql doesn't see stale state from prior turn
    return {"sql": result.sql, "sql_error": None}


def validate_sql(state: AgentState) -> dict:
    """Run sql_guard. Pass canonicalized SQL on success; route to retry on failure."""
    try:
        canonical = guard(state["sql"])
        return {"sql": canonical, "sql_error": None}
    except ValueError as e:
        return {
            "sql_error": f"SQL guard rejected the query: {e}",
            "retries": state["retries"] + 1,
        }


def execute_sql(state: AgentState) -> dict:
    """Execute the validated SQL. Route errors back for retry with specifics."""
    try:
        rows = run_query(state["sql"])
        return {"rows": rows, "sql_error": None}

    except OperationalError as e:
        msg = str(e.orig) if e.orig else str(e)
        if "statement timeout" in msg.lower():
            return {
                "sql_error": (
                    "Query exceeded 10 second timeout. "
                    "Rewrite to be more selective — add WHERE clauses, "
                    "reduce joins, or aggregate earlier."
                ),
                "retries": state["retries"] + 1,
            }
        return {
            "sql_error": f"Database connection error: {msg}",
            "retries": state["retries"] + 1,
        }

    except DBAPIError as e:
        msg = str(e.orig) if e.orig else str(e)
        return {
            "sql_error": msg,
            "retries": state["retries"] + 1,
        }

    except Exception as e:
        return {
            "sql_error": f"Unexpected error: {type(e).__name__}: {e}",
            "retries": state["retries"] + 1,
        }


def _force_default_spec(rows: list[dict], question: str) -> ChartSpec:
    """Pick a sensible default chart when the user explicitly asked for one
    but the LLM declined. Heuristic: take the rightmost numeric column as y
    (typically the aggregate) and the first non-numeric or first column as x."""
    columns = list(rows[0].keys())
    numeric_cols = [
        c for c in columns
        if all(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)
               for r in rows[:5] if r.get(c) is not None)
    ]
    y = numeric_cols[-1] if numeric_cols else columns[-1]
    x_candidates = [c for c in columns if c != y]
    x = x_candidates[0] if x_candidates else columns[0]
    return ChartSpec(
        kind="bar", x=x, y=y,
        title=question.strip().rstrip("?.!")[:80],
        reasoning="User explicitly requested a chart; picked bar as a sensible default.",
    )


def generate_chart(state: AgentState) -> dict:
    """Pick a visualization for the result rows. Runs after summarize on the data path.

    Three modes driven by the user's raw question:
      - "skip"  — user explicitly said no chart → return None, no LLM call
      - "force" — user explicitly asked for a chart → LLM must pick one;
                  if it picks "none" anyway, fall back to a deterministic default
      - "auto"  — silent on charts → current heuristic (LLM picks; pre-filter
                  out trivial cases like single rows)
    """
    rows = state.get("rows") or []

    if state.get("sql_error") or not rows:
        return {"chart": None}

    # Explicit kind override (set by front_agent on the rechart path) wins —
    # deterministic, no LLM call. "table" means the user wants no chart.
    override = state.get("chart_kind_override")
    if override:
        if override == "table":
            return {"chart": None}
        # On rechart, the prior chart's x/y/title are the right source —
        # the title was crafted as a descriptive caption for THIS data on
        # the original turn. The current turn's `question` is literally
        # the rechart command ("replot as pie") — useless as a caption.
        prior = state.get("chart")
        if prior and prior.x and prior.y:
            spec = ChartSpec(
                kind=override,
                x=prior.x,
                y=prior.y,
                title=prior.title or _force_default_spec(
                    rows, state.get("data_question") or state["question"]
                ).title,
                reasoning=f"User requested a {override} chart; reusing prior axes/title.",
            )
        else:
            # No prior chart (first-turn force, or prior was unchartable).
            # Use the original meaningful question, never the rechart command.
            spec = _force_default_spec(
                rows, state.get("data_question") or state["question"]
            )
            spec.kind = override
            spec.reasoning = f"User explicitly requested a {override} chart."
        return {"chart": spec}

    raw_question = state.get("question") or ""
    directive = detect_chart_directive(raw_question)

    if directive == "skip":
        return {"chart": None}

    # Auto-mode skips trivially small results. Force-mode does not — if the
    # user asked for a chart of one row, we'll still try.
    if directive == "auto" and len(rows) == 1:
        return {"chart": None}

    columns = list(rows[0].keys())
    sample = rows[:CHART_ROW_BUDGET]
    user = CHART_SPEC_USER.format(
        directive_note=CHART_FORCE_NOTE if directive == "force" else "",
        question=state.get("data_question") or state["question"],
        columns=", ".join(columns),
        n_total=len(rows),
        n_shown=len(sample),
        rows_json=json.dumps(sample, default=str, indent=2),
    )
    spec: ChartSpec = _chart_picker().invoke(
        [{"role": "system", "content": CHART_SPEC_SYSTEM}, {"role": "user", "content": user}]
    )

    # Validate the LLM's column choices against the actual result columns —
    # the prompt forbids invented names, but verify deterministically.
    if spec.kind in ("bar", "line", "pie"):
        if spec.x not in columns or spec.y not in columns:
            if directive == "force":
                spec = _force_default_spec(rows, state.get("data_question") or state["question"])
            else:
                return {"chart": None}

    # Force override: if the LLM still picked "none"/"table" despite the
    # explicit instruction, coerce to a default chart.
    if spec.kind in ("none", "table") and directive == "force":
        spec = _force_default_spec(rows, state.get("data_question") or state["question"])

    if spec.kind == "none":
        return {"chart": None}
    return {"chart": spec}


async def summarize(state: AgentState) -> dict:
    """Async + ainvoke so token-level streaming events propagate when the graph
    is invoked under astream(stream_mode='messages')."""
    question = state.get("data_question") or state["question"]
    system = SUMMARIZE_SYSTEM.format(dataset_notes=DATASET_NOTES)
    user = SUMMARIZE_USER.format(
        question=question,
        sql=state.get("sql") or "(no SQL was generated)",
        rows_block=_rows_block(state.get("rows"), state.get("sql_error")),
    )
    msg = await _summarizer().ainvoke(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    summary_text = msg.content if isinstance(msg.content, str) else "".join(
        b.get("text", "") for b in msg.content
        if isinstance(b, dict) and b.get("type") in ("text", "text_delta")
    )
    return {
        "summary": summary_text,
        "messages": [AIMessage(content=summary_text)],
    }
