import asyncio
import json
from functools import cache

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.exc import DBAPIError, OperationalError

from agent.db import (
    pagila_schema_string,
    pagila_table_index_string,
    run_query,
    vocabulary_string,
)
from agent.chart_directive import detect as detect_chart_directive
from agent.prompts import (
    CHART_CODE_SYSTEM,
    CHART_CODE_USER,
    CHART_FORCE_NOTE,
    DATASET_NOTES,
    DIAGNOSE_EMPTY_SYSTEM,
    DIAGNOSE_EMPTY_USER,
    GENERATE_RESPONSE_SYSTEM,
    GENERATE_RESPONSE_USER,
    RECHART_USER,
    FRONT_AGENT_SYSTEM,
    FRONT_AGENT_USER,
    REPORT_AGGREGATOR_SYSTEM,
    REPORT_AGGREGATOR_USER,
    REPORT_PLANNER_SYSTEM,
    REPORT_PLANNER_USER,
    SECTION_SUMMARIZER_SYSTEM,
    SECTION_SUMMARIZER_USER,
    SQL_GENERATION_RETRY_HINT,
    SQL_GENERATION_SYSTEM,
    SQL_GENERATION_USER,
    SUMMARIZE_SYSTEM,
    SUMMARIZE_USER,
)
from agent.safety import check_input
from agent.schemas import FrontAgentDecision, ReportPlan, ReportSection, SQLGeneration
from agent.sql_guard import guard
from agent.state import AgentState, ChartCode

MAX_RETRIES = 2
SUMMARIZE_ROW_BUDGET = 50
HISTORY_TURN_LIMIT = 12   # last N messages shown to front agent
CHART_ROW_BUDGET = 20     # rows shown to chart picker — enough to see shape, cheap to send

# LLM transport hardening. Three layers, defense-in-depth:
#   1. SDK request timeout (LLM_REQUEST_TIMEOUT_S): per-attempt cap on the
#      HTTP roundtrip. Anthropic's default is 600s — way too long; a slow
#      response shouldn't make a user wait 10 minutes for an error.
#   2. SDK max_retries: auto-retries 429s, 5xx, and connection errors with
#      backoff. Two retries → 3 attempts max → ≤90s worst-case per call.
#   3. Hard async ceiling (LLM_HARD_TIMEOUT_S): asyncio.wait_for around the
#      whole call. Belt-and-suspenders for socket/DNS stalls that bypass
#      the SDK's HTTP timer (rare, but observed in production). Without
#      this, a user-visible "thinking…" can hang forever.
LLM_REQUEST_TIMEOUT_S = 30.0
LLM_MAX_RETRIES = 2
LLM_HARD_TIMEOUT_S = 90.0   # > LLM_REQUEST_TIMEOUT_S * (LLM_MAX_RETRIES + 1)


def _make_llm(model: str) -> ChatAnthropic:
    """Single source of truth for transport config across all nodes."""
    return ChatAnthropic(
        model=model,
        temperature=0,
        timeout=LLM_REQUEST_TIMEOUT_S,
        max_retries=LLM_MAX_RETRIES,
    )


class LLMTransportError(Exception):
    """Raised when an LLM call exceeds the hard async ceiling or transport
    fails after SDK retries. Caught at the node level and converted into a
    graceful state update so the graph never freezes the UI."""


async def _safe_ainvoke(runnable, messages, *, op: str, hard_timeout: float = LLM_HARD_TIMEOUT_S):
    """Async LLM call with a hard wall-clock ceiling.
    Raises LLMTransportError on timeout or exhausted retries — caller
    decides how to degrade (cached prior output, terse fallback string)."""
    try:
        return await asyncio.wait_for(runnable.ainvoke(messages), timeout=hard_timeout)
    except asyncio.TimeoutError as e:
        raise LLMTransportError(f"{op} timed out after {hard_timeout:.0f}s") from e
    except Exception as e:
        raise LLMTransportError(f"{op} failed: {type(e).__name__}: {e}") from e


async def _safe_invoke_in_thread(runnable, messages, *, op: str, hard_timeout: float = LLM_HARD_TIMEOUT_S):
    """Sync `.invoke()` offloaded to a worker thread, with a hard ceiling.
    Used by `sub_query` so a stalled SDK call in one parallel branch can't
    block its siblings indefinitely. The thread itself isn't killable
    (Python limitation) but the await releases — the orphan thread will
    eventually exit when the SDK does."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(runnable.invoke, messages),
            timeout=hard_timeout,
        )
    except asyncio.TimeoutError as e:
        raise LLMTransportError(f"{op} timed out after {hard_timeout:.0f}s") from e
    except Exception as e:
        raise LLMTransportError(f"{op} failed: {type(e).__name__}: {e}") from e


def _cached_system(text: str) -> dict:
    """System message with Anthropic prompt caching enabled.

    Marks the entire system prompt as cacheable. First call within a 5-minute
    window pays a ~25% write premium on cached tokens; subsequent reads are
    ~10% of normal input cost. Cache hits show up as `cache_read_input_tokens`
    in the response usage metadata.

    Only worth using on prompts >= 1024 tokens (Anthropic's caching minimum) —
    smaller prompts ignore the cache_control marker. Used here on the
    FRONT_AGENT and SQL_GENERATION system prompts (both well over threshold).
    The schema string alone is ~6.7K tokens; caching it eliminates the
    dominant repeat cost on every data-path turn.
    """
    return {
        "role": "system",
        "content": [
            {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}},
        ],
    }


@cache
def _front_agent_llm():
    return _make_llm("claude-sonnet-4-6").with_structured_output(FrontAgentDecision)


@cache
def _sql_generator():
    return _make_llm("claude-sonnet-4-6").with_structured_output(SQLGeneration)


@cache
def _summarizer():
    # Plain LLM (no structured output) so token streaming has clean text chunks
    return _make_llm("claude-haiku-4-5-20251001")


@cache
def _response_llm():
    # Plain LLM for the streaming respond/rechart reply. Haiku — short outputs,
    # latency matters here since this is the user's first visible token after
    # routing.
    return _make_llm("claude-haiku-4-5-20251001")


@cache
def _chart_picker():
    return _make_llm("claude-sonnet-4-6").with_structured_output(ChartCode)


@cache
def _report_planner():
    return _make_llm("claude-sonnet-4-6").with_structured_output(ReportPlan)


@cache
def _section_summarizer():
    # Plain Haiku — short per-section blurbs, fast.
    return _make_llm("claude-haiku-4-5-20251001")


@cache
def _report_aggregator():
    # Plain Sonnet — composing the report is the user-visible final pass.
    # Sonnet's prose quality matters more than latency here.
    return _make_llm("claude-sonnet-4-6")


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
        [_cached_system(system), {"role": "user", "content": user}]
    )

    if decision.intent == "data":
        return {"intent": "data", "data_question": decision.data_question}

    if decision.intent == "rechart":
        # No summary/messages here — generate_response will stream the reply.
        return {
            "intent": "rechart",
            "chart_kind_override": decision.chart_kind_override,
        }

    if decision.intent == "report":
        # Report path: plan_report → fan-out → aggregate. No summary set here.
        return {"intent": "report"}

    # respond — leave summary unset so generate_response can stream it.
    return {"intent": "respond"}


async def generate_response(state: AgentState) -> dict:
    """Stream a brief plain-text reply for respond/rechart intents.

    Async + ainvoke so token-level streaming events propagate under
    astream_events. The streaming chunks land directly in the user-facing
    cl.Message (filtered on `langgraph_node == "generate_response"` in app.py).
    Sets state.summary on completion as a defensive fallback if streaming
    chunks were lost.
    """
    intent = state["intent"]

    if intent == "rechart":
        prior = state.get("chart")
        kind_hint = state.get("chart_kind_override") or "(not specified)"
        intent_context = (
            "User wants to modify the prior chart.\n"
            f"Requested chart kind: {kind_hint}\n"
            f"Prior chart title: {prior.title if prior else '(none)'}\n"
        )
    else:
        intent_context = ""

    user_msg = GENERATE_RESPONSE_USER.format(
        conversation=_format_history(state.get("messages") or [], state["question"]),
        question=state["question"],
        intent=intent,
        intent_context=intent_context,
    )
    try:
        msg = await _safe_ainvoke(
            _response_llm(),
            [{"role": "system", "content": GENERATE_RESPONSE_SYSTEM},
             {"role": "user", "content": user_msg}],
            op="generate_response",
        )
        text = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content
            if isinstance(b, dict) and b.get("type") in ("text", "text_delta")
        )
    except LLMTransportError as e:
        text = (
            "Sorry — I couldn't reach the language model just now "
            f"({e}). Please try again in a moment."
        )
    return {
        "summary": text,
        "messages": [AIMessage(content=text)],
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
        vocabulary=vocabulary_string(),
        schema=pagila_schema_string(),
    )
    user = SQL_GENERATION_USER.format(retry_context=retry_context, question=question)

    # System prompt (incl. ~6.7K-token schema) is fully stable across calls →
    # cache it. Variable retry context lives in the user message, uncached.
    result: SQLGeneration = _sql_generator().invoke(
        [_cached_system(system), {"role": "user", "content": user}]
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


def _generate_chart_code_for(question: str, rows: list[dict], force: bool = False) -> ChartCode | None:
    """Ask the LLM to write Plotly code for these rows.

    Returns a ChartCode (with `.code` populated) when the LLM produces one,
    or None when the LLM declines (no chart fits) and `force` is False.
    Doesn't execute the code — that happens in app._build_chart against
    the sandbox. Empty/None code → None.
    """
    if not rows:
        return None
    columns = list(rows[0].keys())
    sample = rows[:CHART_ROW_BUDGET]
    user = CHART_CODE_USER.format(
        directive_note=CHART_FORCE_NOTE if force else "",
        question=question,
        columns=", ".join(columns),
        n_total=len(rows),
        n_shown=len(sample),
        rows_json=json.dumps(sample, default=str, indent=2),
    )
    chart_code: ChartCode = _chart_picker().invoke(
        [{"role": "system", "content": CHART_CODE_SYSTEM},
         {"role": "user", "content": user}]
    )
    if not chart_code.code:
        return None
    return chart_code


def generate_chart(state: AgentState) -> dict:
    """Generate Python plotting code for the result rows.

    Two flows depending on whether this is a fresh data turn or a rechart turn:

    - **Data turn**: LLM writes code from scratch given the question + rows.
      Honors the user-message chart directive (skip / force / auto).
    - **Rechart turn**: LLM gets the prior code + the user's modification
      request and produces a new version — arbitrary edits supported
      ("switch x", "color by genre", "log scale on y", "rotate labels", ...).

    Two deterministic fast paths skip the LLM entirely:
    - sql_error or no rows → no chart
    - chart_kind_override == "table" → user wants no chart at all
    """
    rows = state.get("rows") or []
    if state.get("sql_error") or not rows:
        return {"chart": None}

    intent = state.get("intent")
    override = state.get("chart_kind_override")

    if override == "table":
        return {"chart": None}

    columns = list(rows[0].keys())
    sample = rows[:CHART_ROW_BUDGET]

    # ── Rechart path ──────────────────────────────────────────────
    if intent == "rechart":
        prior = state.get("chart")
        user = RECHART_USER.format(
            data_question=state.get("data_question") or "(unknown)",
            question=state.get("question") or "(no message)",
            kind_hint=override or "(none)",
            prior_title=prior.title if prior else "(none)",
            prior_reasoning=prior.reasoning if prior else "(none)",
            prior_code=prior.code if prior and prior.code else "(no prior code — generate fresh)",
            columns=", ".join(columns),
            n_total=len(rows),
            n_shown=len(sample),
            rows_json=json.dumps(sample, default=str, indent=2),
        )
        chart_code: ChartCode = _chart_picker().invoke(
            [{"role": "system", "content": CHART_CODE_SYSTEM},
             {"role": "user", "content": user}]
        )
        if not chart_code.code:
            # LLM declined to produce code — preserve prior chart rather than
            # erasing the user's existing visualization.
            return {"chart": prior if prior else None}
        return {"chart": chart_code}

    # ── Data turn path ────────────────────────────────────────────
    raw_question = state.get("question") or ""
    directive = detect_chart_directive(raw_question)

    if directive == "skip":
        return {"chart": None}
    if directive == "auto" and len(rows) == 1:
        return {"chart": None}

    chart_code = _generate_chart_code_for(
        question=state.get("data_question") or state["question"],
        rows=rows,
        force=(directive == "force"),
    )
    return {"chart": chart_code}


def diagnose_empty(state: AgentState) -> dict:
    """Run a diagnostic when execute_sql succeeded with 0 rows.

    Generates a follow-up SQL via the SQL-gen LLM (different prompt) that
    surfaces what values DO exist in the data the user was filtering on.
    Goes through sql_guard like any other query — read-only fence preserved.
    Failures (LLM error, guard rejection, executor error) degrade silently:
    summarize handles missing diagnostic gracefully.
    """
    # Only run for "real empty" — sql_error or rows already present means skip.
    if state.get("sql_error") or state.get("rows"):
        return {"diagnostic_sql": None, "diagnostic_rows": None}

    user = DIAGNOSE_EMPTY_USER.format(
        question=state.get("data_question") or state["question"],
        sql=state.get("sql") or "",
        vocabulary=vocabulary_string(),
        schema=pagila_schema_string(),
    )
    try:
        diag: SQLGeneration = _sql_generator().invoke(
            [_cached_system(DIAGNOSE_EMPTY_SYSTEM),
             {"role": "user", "content": user}]
        )
    except Exception:
        return {"diagnostic_sql": None, "diagnostic_rows": None}

    try:
        canonical = guard(diag.sql)
    except ValueError:
        return {"diagnostic_sql": None, "diagnostic_rows": None}

    try:
        rows = run_query(canonical)
    except Exception:
        return {"diagnostic_sql": None, "diagnostic_rows": None}

    # Cap diagnostic rows for prompt cost
    return {"diagnostic_sql": canonical, "diagnostic_rows": rows[:20]}


def _diagnostic_block(diag_sql: str | None, diag_rows: list[dict] | None) -> str:
    """Markdown-formatted diagnostic block to inject into SUMMARIZE_USER.
    Empty string when there's no diagnostic to surface."""
    if not diag_sql or diag_rows is None:
        return ""
    if not diag_rows:
        return "\n\nDiagnostic ran but returned no rows either — the data may be empty more broadly."
    payload = json.dumps(diag_rows, default=str, indent=2)
    return (
        "\n\nThe main query returned 0 rows. We ran a diagnostic to surface "
        "what values DO have data in this context:\n\n"
        f"```sql\n{diag_sql}\n```\n\n"
        f"Diagnostic results ({len(diag_rows)} rows):\n```json\n{payload}\n```"
    )


async def summarize(state: AgentState) -> dict:
    """Async + ainvoke so token-level streaming events propagate when the graph
    is invoked under astream(stream_mode='messages')."""
    question = state.get("data_question") or state["question"]
    system = SUMMARIZE_SYSTEM.format(dataset_notes=DATASET_NOTES)
    user = SUMMARIZE_USER.format(
        question=question,
        sql=state.get("sql") or "(no SQL was generated)",
        rows_block=_rows_block(state.get("rows"), state.get("sql_error")),
        diagnostic_block=_diagnostic_block(
            state.get("diagnostic_sql"), state.get("diagnostic_rows")
        ),
    )
    try:
        msg = await _safe_ainvoke(
            _summarizer(),
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            op="summarize",
        )
        summary_text = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content
            if isinstance(b, dict) and b.get("type") in ("text", "text_delta")
        )
    except LLMTransportError as e:
        # Surface a useful failure rather than freezing the user's chat.
        # Include the SQL row count so they at least know the query worked.
        rows = state.get("rows") or []
        summary_text = (
            f"Got {len(rows)} row(s) back, but the summarizer is unreachable "
            f"({e}). The query and results are visible in the steps above."
        )
    return {
        "summary": summary_text,
        "messages": [AIMessage(content=summary_text)],
    }


# ── Report mode ──────────────────────────────────────────────────────────────

REPORT_SECTION_ROW_PREVIEW = 10
REPORT_SECTION_ROW_BUDGET = 30   # rows shown to the section summarizer LLM
REPORT_SECTION_CHART_ROW_BUDGET = 500  # rows persisted on the section for chart rendering


async def warmup_sql_cache(state: AgentState) -> dict:
    """Pre-warm the SQL-generator's prompt cache before parallel fan-out.

    Anthropic's prompt cache is shared across requests, but only after a
    write completes. When N parallel sub_queries fire at once, none of them
    see the cache yet — they all pay the 1.25× write premium on the same
    ~8.9K-token SQL-gen prefix (schema + dataset notes + vocabulary). One
    serialized call writes the cache; the subsequent fan-out hits it.

    Trade-off: +1 LLM call (~$0.02, ~2-3s) before the report fans out.
    Saves on a 5-section report: ~30K input tokens billed at write rate
    (≈ 4 × 8.9K × 1.15 redundant write premium).

    Failures are silent — a warmup miss only forfeits the optimization;
    each sub_query will write its own cache as before.
    """
    sys_prompt = SQL_GENERATION_SYSTEM.format(
        dataset_notes=DATASET_NOTES,
        vocabulary=vocabulary_string(),
        schema=pagila_schema_string(),
    )
    usr_prompt = SQL_GENERATION_USER.format(
        retry_context="",
        question="(cache warmup — produce any short valid SELECT)",
    )
    try:
        await _safe_invoke_in_thread(
            _sql_generator(),
            [_cached_system(sys_prompt), {"role": "user", "content": usr_prompt}],
            op="warmup_sql_cache",
            hard_timeout=20.0,  # tight ceiling — if warmup is slow, give up
        )
    except Exception:
        pass  # cache miss on fan-out is the worst case; never block the report
    return {}


def plan_report(state: AgentState) -> dict:
    """Decompose the user's broad question into 3-7 sub-questions.

    Resets the report_sections accumulator (via the custom reducer's None
    sentinel) so prior-turn sections don't leak in.
    """
    user = REPORT_PLANNER_USER.format(
        question=state.get("data_question") or state["question"],
        conversation=_format_history(state.get("messages") or [], state["question"]),
    )
    # Planner gets the compact table index (~1K tokens), NOT the full DDL +
    # sample rows used by the SQL generator. The planner only emits natural-
    # language sub-questions; it doesn't need column types or vocabulary.
    # That drops the planner's prompt from ~9.9K to ~2K tokens — small enough
    # that prompt caching isn't worth it. Reports rarely happen back-to-back
    # within Anthropic's 5-min cache TTL, so the cache-write premium (1.25×)
    # would dominate vs paying 1× per call.
    system = REPORT_PLANNER_SYSTEM.format(
        dataset_notes=DATASET_NOTES,
        table_index=pagila_table_index_string(),
    )
    plan: ReportPlan = _report_planner().invoke(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}]
    )
    return {
        "report_outline": plan.sections,
        "report_plan_rationale": plan.rationale,
        "report_sections": None,  # reset accumulator via custom reducer
    }


def _section_chart(section: ReportSection, rows: list[dict]) -> ChartCode | None:
    """Generate Plotly code for ONE report section.

    Calls the LLM (one extra call per section — the cost trade-off the
    user accepted in exchange for chart quality on par with what Claude
    Chat produces). The planner's `chart_hint` is folded into the question
    so the chart-coder picks the same kind unless the data argues otherwise.
    Returns None if the planner asked for "none"/"table", if the LLM
    declines, or if the row count is too small to plot meaningfully.
    """
    if not rows:
        return None
    if section.chart_hint in ("none", "table"):
        return None
    # Pass the planner's hint as part of the question so the chart-coder
    # has the same context the planner did. Force=True since the planner
    # already opted in to a chart for this section.
    chart_request = (
        f"{section.sub_question}\n"
        f"(Section title: '{section.title}'. Suggested chart kind: '{section.chart_hint}'.)"
    )
    return _generate_chart_code_for(question=chart_request, rows=rows, force=True)


async def sub_query(state: AgentState) -> dict:
    """Run the SQL pipeline for ONE report section, in parallel with siblings.

    Each invocation receives a `current_section` payload via Send. Failures
    in any step degrade gracefully — the section keeps its title and gets
    an `error` field so the aggregator can mention what couldn't be answered.

    All blocking I/O (LLM .invoke, run_query) is offloaded to a worker thread
    via asyncio.to_thread so parallel Send branches don't serialize on the
    event loop — without this, on_chain_start events for sibling branches
    can't flush until the running branch hits its first await.
    """
    section: ReportSection | None = state.get("current_section")
    if section is None:
        return {}  # defensive — Send should always populate this

    # Helper: emit a degraded section with an error and stop
    def _failed(msg: str) -> dict:
        return {"report_sections": [section.model_copy(update={"error": msg})]}

    # 1. Generate SQL — uses _safe_invoke_in_thread so a stalled SDK call
    # in this branch can't block sibling sections indefinitely.
    try:
        sys_prompt = SQL_GENERATION_SYSTEM.format(
            dataset_notes=DATASET_NOTES,
            vocabulary=vocabulary_string(),
            schema=pagila_schema_string(),
        )
        usr_prompt = SQL_GENERATION_USER.format(retry_context="", question=section.sub_question)
        sql_result: SQLGeneration = await _safe_invoke_in_thread(
            _sql_generator(),
            [_cached_system(sys_prompt), {"role": "user", "content": usr_prompt}],
            op=f"sub_query.sql_gen[{section.title}]",
        )
        raw_sql = sql_result.sql
    except LLMTransportError as e:
        return _failed(f"SQL generation transport error: {e}")
    except Exception as e:
        return _failed(f"SQL generation failed: {type(e).__name__}: {e}")

    # 2. Validate
    try:
        canonical = guard(raw_sql)
    except ValueError as e:
        return _failed(f"Validator rejected SQL: {e}")

    # 3. Execute (sqlalchemy is sync — offload so siblings can run concurrently)
    try:
        rows = await asyncio.to_thread(run_query, canonical)
    except Exception as e:
        return _failed(f"Execution failed: {type(e).__name__}: {str(e)[:200]}")

    # 4. Section summary (1-2 sentences via Haiku, no streaming needed)
    preview = rows[:REPORT_SECTION_ROW_BUDGET]
    summarizer_user = SECTION_SUMMARIZER_USER.format(
        title=section.title,
        sub_question=section.sub_question,
        sql=canonical,
        row_count=len(rows),
        shown_note=f", showing first {len(preview)}" if len(rows) > len(preview) else "",
        rows_preview=json.dumps(preview, default=str, indent=2),
    )
    try:
        summary_msg = await _safe_ainvoke(
            _section_summarizer(),
            [{"role": "system", "content": SECTION_SUMMARIZER_SYSTEM},
             {"role": "user", "content": summarizer_user}],
            op=f"sub_query.summarize[{section.title}]",
        )
        section_summary = (
            summary_msg.content if isinstance(summary_msg.content, str)
            else "".join(
                b.get("text", "") for b in summary_msg.content
                if isinstance(b, dict) and b.get("type") in ("text", "text_delta")
            )
        )
    except LLMTransportError as e:
        section_summary = f"(summary unavailable: {e})"
    except Exception as e:
        section_summary = f"(summary unavailable: {type(e).__name__})"

    # 5. Chart code (LLM call — also offloaded with the hard ceiling so it
    # can't hold up the parallel branch indefinitely on a stalled SDK call).
    try:
        chart = await asyncio.wait_for(
            asyncio.to_thread(_section_chart, section, rows),
            timeout=LLM_HARD_TIMEOUT_S,
        )
    except (asyncio.TimeoutError, Exception):
        chart = None

    completed = section.model_copy(update={
        "sql": canonical,
        "row_count": len(rows),
        "rows_preview": rows[:REPORT_SECTION_ROW_PREVIEW],
        "rows_for_chart": rows[:REPORT_SECTION_CHART_ROW_BUDGET],
        "summary": section_summary,
        "chart": chart,
    })
    return {"report_sections": [completed]}


async def aggregate_report(state: AgentState) -> dict:
    """Compose the final markdown report from all completed sections.

    Sections arrive in the accumulator in the order parallel branches finish
    (non-deterministic). Re-sort to match the planner's outline before
    composing so the narrative reads in the intended order. Streams via
    plain LLM under astream_events.
    """
    outline: list[ReportSection] = state.get("report_outline") or []
    completed: list[ReportSection] = state.get("report_sections") or []

    # Sort completed sections back into outline order
    title_to_pos = {s.title: i for i, s in enumerate(outline)}
    sorted_sections = sorted(
        completed, key=lambda s: title_to_pos.get(s.title, len(outline) + 1)
    )

    section_blocks = []
    failed_lines = []
    for s in sorted_sections:
        if s.error:
            failed_lines.append(f"- **{s.title}** — {s.error}")
            continue
        rows_json = json.dumps(s.rows_preview or [], default=str, indent=2)
        section_blocks.append(
            f"### {s.title}\n"
            f"Sub-question: {s.sub_question}\n"
            f"Row count: {s.row_count}\n"
            f"Section blurb (use this verbatim or paraphrase): {s.summary}\n"
            f"Sample rows:\n```json\n{rows_json}\n```"
        )

    user = REPORT_AGGREGATOR_USER.format(
        question=state["question"],
        plan_rationale=state.get("report_plan_rationale") or "(not provided)",
        sections_block="\n\n".join(section_blocks) or "(no sections completed)",
        failed_sections="\n".join(failed_lines) or "(none)",
    )
    try:
        msg = await _safe_ainvoke(
            _report_aggregator(),
            [{"role": "system", "content": REPORT_AGGREGATOR_SYSTEM},
             {"role": "user", "content": user}],
            op="aggregate_report",
            # Slightly longer ceiling — aggregation is the longest single
            # generation in the graph (multi-section narrative).
            hard_timeout=LLM_HARD_TIMEOUT_S * 1.5,
        )
        text = msg.content if isinstance(msg.content, str) else "".join(
            b.get("text", "") for b in msg.content
            if isinstance(b, dict) and b.get("type") in ("text", "text_delta")
        )
    except LLMTransportError as e:
        # Sections are intact in state — surface a fallback that lists
        # what we have so the user gets value even when aggregation fails.
        completed_titles = [s.title for s in sorted_sections if not s.error]
        text = (
            f"The report aggregator is unreachable ({e}). "
            f"{len(completed_titles)} of {len(sorted_sections)} sections "
            "completed; their queries and blurbs are visible in the steps above."
        )
    return {
        "summary": text,
        "report_text": text,
        "messages": [AIMessage(content=text)],
    }
