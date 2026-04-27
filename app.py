import asyncio
import hmac
import logging
import os
import time
from dotenv import load_dotenv
load_dotenv()  # must come before any imports that read env vars
print("[boot diagnostic] env keys present:", sorted(
    k for k in os.environ
    if k.startswith(("DATABASE_", "ANTHROPIC_", "VOYAGE_", "LANGSMITH_",
                     "APP_", "CHAINLIT_", "PORT", "RAILWAY_"))
))

import chainlit as cl
import pandas as pd
import plotly.express as px
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

from agent.db import warmup
from agent.graph import app_graph
from agent.chart_sandbox import SandboxError, execute_chart_code
from agent.state import ChartCode, turn_input

warmup()


# ─── Auth ─────────────────────────────────────────────────────────────────
# Simple shared-credential password auth. Configure via env:
#   CHAINLIT_AUTH_SECRET — random string for cookie signing (required by
#                          Chainlit when ANY auth is enabled). Generate
#                          with `chainlit create-secret` or `python -c
#                          'import secrets; print(secrets.token_urlsafe(32))'`.
#   APP_USERNAME, APP_PASSWORD — the credentials users enter at the login
#                                screen. Both required; if either is unset
#                                the callback rejects everything (deny by
#                                default — never run with a missing
#                                credential silently letting people in).
#
# This is one shared credential, not a per-user system. Right level of
# friction for a portfolio piece: enough to keep web crawlers out and
# prevent random API-cost burn, low enough that a recruiter can DM you
# for the password and try it. Upgrade to OAuth later if you want.

@cl.password_auth_callback
def _auth_callback(username: str, password: str):
    expected_user = os.getenv("APP_USERNAME") or ""
    expected_pass = os.getenv("APP_PASSWORD") or ""
    # Reject everyone if credentials aren't configured. Better than
    # accidentally deploying with empty-string defaults that anyone could match.
    if not expected_user or not expected_pass:
        logger.warning("APP_USERNAME / APP_PASSWORD not set — rejecting all logins.")
        return None
    # Constant-time comparison so an attacker can't time-attack the password.
    if (hmac.compare_digest(username, expected_user)
            and hmac.compare_digest(password, expected_pass)):
        return cl.User(identifier=username, metadata={"role": "user"})
    return None


# Anthropic emits multi-character chunks per stream event. Pacing the
# rendering at character-level with a small delay smooths the typewriter
# feel — chunky word-bursts become a uniform glide.
SMOOTH_STREAM_CHAR_DELAY_S = 0.005   # ~200 chars/sec
SMOOTH_STREAM_CHUNK_THRESHOLD = 4    # only smooth chunks longer than this


PREVIEW_ROWS = 10


def _format_rows_table(rows: list[dict]) -> str:
    if not rows:
        return "_no rows returned_"
    head = rows[:PREVIEW_ROWS]
    cols = list(head[0].keys())
    md = ["| " + " | ".join(cols) + " |",
          "| " + " | ".join("---" for _ in cols) + " |"]
    for r in head:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    suffix = f"\n\n_showing {len(head)} of {len(rows)} rows_" if len(rows) > len(head) else ""
    return "\n".join(md) + suffix


# Pipeline nodes that get their own cl.Step. summarize is excluded — its prose
# streams directly into the user-facing cl.Message.
#
# To surface a NEW graph node in the UI:
#   1. Add a `name → friendly label` entry to NODE_LABEL.
#   2. Add a `name → lucide icon name` entry to NODE_ICON.
#   3. (Optional) Add a branch in `_step_body` for custom output formatting.
#      If you skip this, the generic fallback renders the state-update fields
#      as a key/value list — informative but unstyled.
NODE_LABEL = {
    "front_agent":      "front agent",
    "retrieve_context": "RAG retrieval",
    "generate_sql":     "SQL generator",
    "validate_sql":     "SQL validator",
    "execute_sql":      "Postgres",
    "diagnose_empty":   "empty-result diagnostic",
    "generate_chart":   "chart designer",
    "plan_report":      "report planner",
    "warmup_sql_cache": "prompt-cache warmup",
    "sub_query":        "section query",
    # aggregate_report is excluded — its prose IS the streaming message.
}

NODE_ICON = {
    "front_agent":      "brain",
    "retrieve_context": "book-open",
    "generate_sql":     "code-2",
    "validate_sql":     "shield-check",
    "execute_sql":      "database",
    "diagnose_empty":   "search-check",
    "generate_chart":   "chart-bar",
    "plan_report":      "list-checks",
    "warmup_sql_cache": "flame",
    "sub_query":        "file-search",
}

# How often to refresh elapsed time. 1s feels live without saturating the
# Chainlit websocket — at 5+ parallel sub_query branches that's still <10
# updates/sec total.
TIMER_TICK_S = 1.0


async def _step_timer(step: "cl.Step", base_label: str, started_at: float) -> None:
    """Tick elapsed time into the step header until cancelled.

    The "Using → Used" verb prefix is rendered by Chainlit itself based on
    whether `step.end` is set. We only need to keep `step.end` cleared while
    the node runs (it is, since we never set it on start) and append the
    elapsed-time suffix to the name. on_chain_end will set step.end and
    Chainlit's UI will flip the verb automatically.

    Cancellation is normal control flow here, not an error.
    """
    try:
        while True:
            await asyncio.sleep(TIMER_TICK_S)
            elapsed = time.perf_counter() - started_at
            step.name = f"{base_label}  ·  {elapsed:.0f}s"
            await step.update()
    except asyncio.CancelledError:
        pass


def _utc_now_iso() -> str:
    """Match Chainlit's internal utc_now format so the UI renders cleanly."""
    from chainlit.utils import utc_now
    return utc_now()


def _step_body(node: str, output: dict) -> str:
    if node == "front_agent":
        intent = output.get("intent")
        if intent == "data":
            dq = output.get("data_question") or "(none)"
            return f"_Refined as a data question:_\n\n> {dq}"
        if intent == "rechart":
            kind = output.get("chart_kind_override") or "(picker decides)"
            return f"_Re-charting prior result. Requested kind: `{kind}`._"
        if intent == "report":
            return "_Routed to report generation — planning multi-section report._"
        # respond — either a deterministic refusal (summary already set) or a
        # routing-only decision that generate_response will fill in.
        if output.get("summary"):
            return "_Routed to direct reply (refused at the safety gate)._"
        return "_Routed to direct reply — no SQL needed._"
    if node == "generate_sql":
        sql = output.get("sql") or "(no SQL produced)"
        return f"```sql\n{sql}\n```"
    if node == "validate_sql":
        if output.get("sql_error"):
            return f"❌ {output['sql_error']}"
        return "_Passed all read-only checks._"
    if node == "execute_sql":
        if output.get("sql_error"):
            return f"❌ {output['sql_error']}"
        rows = output.get("rows") or []
        body = f"Returned **{len(rows)}** row(s)."
        if rows:
            body += f"\n\n{_format_rows_table(rows)}"
        return body
    if node == "diagnose_empty":
        return _diagnose_step_body(output)
    if node == "generate_chart":
        return _chart_step_body(output.get("chart"))
    if node == "retrieve_context":
        return _retrieve_step_body(output)
    if node == "plan_report":
        return _plan_report_step_body(output)
    if node == "warmup_sql_cache":
        return (
            "_Pre-warming Anthropic prompt cache so the parallel section "
            "queries below all hit the cache instead of each paying the "
            "write premium on the ~9K-token SQL system prompt._"
        )
    if node == "sub_query":
        return _sub_query_step_body(output)
    # Fallback for any node added to NODE_LABEL without a custom branch above.
    return _generic_step_body(output)


def _retrieve_step_body(output: dict) -> str:
    """Step body for retrieve_context — list what was retrieved + scores
    so the user can see *why* the SQL generator made certain choices."""
    glossary = output.get("retrieved_glossary") or []
    examples = output.get("retrieved_examples") or []
    if not glossary and not examples:
        return "_No relevant entries above the similarity threshold — using static prompt only._"
    lines = []
    if glossary:
        lines.append(f"**Glossary** ({len(glossary)} hit{'s' if len(glossary) != 1 else ''}):")
        for h in glossary:
            sim = h.get("similarity") or 0.0
            lines.append(f"- `{h['id']}` — sim {sim:.2f}")
    if examples:
        lines.append("")
        lines.append(f"**Examples** ({len(examples)} hit{'s' if len(examples) != 1 else ''}):")
        for h in examples:
            sim = h.get("similarity") or 0.0
            lines.append(f"- `{h['id']}` — sim {sim:.2f}")
    return "\n".join(lines)


def _plan_report_step_body(output: dict) -> str:
    """Step body for plan_report — show the planned outline + rationale."""
    outline = output.get("report_outline") or []
    rationale = output.get("report_plan_rationale") or ""
    if not outline:
        return "_No plan produced._"
    lines = [f"_{rationale}_", "", f"**{len(outline)} sections planned:**", ""]
    for i, section in enumerate(outline, 1):
        # ReportSection objects — access fields directly
        title = getattr(section, "title", "(untitled)")
        sub_q = getattr(section, "sub_question", "")
        hint = getattr(section, "chart_hint", None)
        hint_str = f" · `{hint}`" if hint else ""
        lines.append(f"{i}. **{title}**{hint_str} — {sub_q}")
    return "\n".join(lines)


def _sub_query_step_body(output: dict) -> str:
    """Step body for sub_query — show the section's outcome compactly."""
    sections = output.get("report_sections") or []
    if not sections:
        return "_(no section produced)_"
    s = sections[0]  # each sub_query appends exactly one
    title = getattr(s, "title", "(untitled)")
    if getattr(s, "error", None):
        return f"**{title}** — ❌ {s.error}"
    sql = getattr(s, "sql", None) or "(no SQL)"
    row_count = getattr(s, "row_count", 0)
    summary = getattr(s, "summary", None) or "(no summary)"
    body = f"**{title}** — {row_count} row(s)\n\n_{summary}_\n\n```sql\n{sql}\n```"
    return body


def _diagnose_step_body(output: dict) -> str:
    """Step body for diagnose_empty. Shows the diagnostic SQL + its rows so
    the user can see exactly how the agent investigated the empty result."""
    sql = output.get("diagnostic_sql")
    rows = output.get("diagnostic_rows")
    if not sql:
        return "_No diagnostic was produced — query failed too quickly to investigate, or the LLM declined._"
    body = (
        "Main query returned 0 rows. Ran a diagnostic to surface what values "
        "DO have data:\n\n"
        f"```sql\n{sql}\n```"
    )
    if rows is not None:
        body += f"\n\n**{len(rows)}** diagnostic row(s)."
        if rows:
            body += f"\n\n{_format_rows_table(rows)}"
    return body


def _generic_step_body(output: dict) -> str:
    """Default formatter for any graph node that doesn't have a dedicated
    branch above. Renders non-null state-update fields as a compact list.
    Means a new node added to NODE_LABEL gets visible content immediately,
    even before custom formatting is written."""
    if not output:
        return "_(no output)_"
    parts = []
    for k, v in output.items():
        if v is None:
            continue
        if isinstance(v, bool):
            parts.append(f"- **{k}**: `{v}`")
        elif isinstance(v, (int, float)):
            parts.append(f"- **{k}**: `{v}`")
        elif isinstance(v, str):
            preview = v if len(v) <= 200 else v[:200] + "…"
            parts.append(f"- **{k}**: `{preview}`")
        elif isinstance(v, list):
            parts.append(f"- **{k}**: list[{len(v)}]")
        elif isinstance(v, dict):
            parts.append(f"- **{k}**: dict[{len(v)}]")
        else:
            parts.append(f"- **{k}**: `{type(v).__name__}`")
    return "\n".join(parts) if parts else "_(no fields set)_"


def _chart_step_body(chart) -> str:
    """Markdown body for the generate_chart Step."""
    if chart is None:
        return "_No chart for this result._"
    lines = []
    if chart.title:
        lines.append(f"**{chart.title}**")
    if chart.reasoning:
        lines.append(f"_{chart.reasoning}_")
    if chart.code:
        lines.append(f"```python\n{chart.code}\n```")
    return "\n\n".join(lines) if lines else "_(chart code produced)_"


async def _stream_smoothed(msg: cl.Message, text: str) -> None:
    """Pace token rendering at character-level for smoother feel.
    Anthropic streams multi-char deltas — passing them straight to stream_token
    looks bursty. Splitting into chars with a tiny delay yields a steadier flow.
    Short chunks pass through directly to avoid pointless overhead."""
    if len(text) <= SMOOTH_STREAM_CHUNK_THRESHOLD:
        await msg.stream_token(text)
        return
    for ch in text:
        await msg.stream_token(ch)
        await asyncio.sleep(SMOOTH_STREAM_CHAR_DELAY_S)


def _build_chart(chart: ChartCode | None, rows: list[dict], context: str = ""):
    """Execute LLM-generated chart code in the sandbox; return the produced Figure.

    Graceful degradation: any sandbox/runtime failure logs to stderr and returns
    None so the prose summary still goes through. Logging matters: a silently
    skipped chart looks identical to a section that legitimately had no chart.
    """
    if chart is None or not chart.code:
        return None
    label = f" [{context}]" if context else ""
    try:
        return execute_chart_code(chart.code, rows)
    except SandboxError as e:
        logger.warning("Chart sandbox rejected code%s: %s", label, e)
        logger.warning("Offending code:\n%s", chart.code)
        return None
    except Exception as e:
        logger.warning("Chart build raised%s: %s: %s", label, type(e).__name__, e)
        return None


def _extract_text(chunk_content) -> str:
    if isinstance(chunk_content, str):
        return chunk_content
    if isinstance(chunk_content, list):
        parts = []
        for b in chunk_content:
            if isinstance(b, dict):
                if b.get("type") in ("text", "text_delta") and "text" in b:
                    parts.append(b["text"])
                elif "text" in b and isinstance(b["text"], str):
                    parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return ""


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", cl.context.session.id)
    await cl.Message(
        content="Ready. Ask me anything about the Pagila database (DVD rental store, 2022 data)."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    # LangSmith picks up `tags` and `metadata` automatically when
    # LANGSMITH_TRACING=true. Tagging by source lets you filter the dashboard
    # for "user chats" vs "eval runs" without trawling a single firehose.
    config = {
        "configurable": {"thread_id": thread_id},
        "tags": ["chainlit"],
        "metadata": {"source": "chainlit", "thread_id": thread_id},
        "run_name": f"chat:{(message.content or '')[:60]}",
    }

    # Each entry tracks the cl.Step, the live-timer task that updates it
    # while the node runs, and the start time for elapsed-time calculation.
    open_steps: dict[str, dict] = {}
    final_answer = cl.Message(content="")
    summary_from_state: str | None = None
    rows_for_chart: list[dict] = []
    chart_spec: ChartCode | None = None
    fa_intent: str | None = None  # captured from front_agent's on_chain_end

    try:
        async for ev in app_graph.astream_events(
            turn_input(message.content, HumanMessage(content=message.content)),
            config=config,
            version="v2",
        ):
            kind = ev["event"]
            name = ev.get("name")
            meta = ev.get("metadata") or {}
            node_for_metadata = meta.get("langgraph_node")

            # Stream LLM tokens for nodes whose output IS the visible message:
            #   - summarize        (data path)
            #   - generate_response (respond/rechart paths)
            #   - aggregate_report (report path — the multi-section narrative)
            # No Step for any of these — the streaming message IS their output.
            if (
                kind == "on_chat_model_stream"
                and node_for_metadata in ("summarize", "generate_response", "aggregate_report")
            ):
                chunk = ev["data"].get("chunk")
                if chunk is not None:
                    text = _extract_text(chunk.content)
                    if text:
                        await _stream_smoothed(final_answer, text)
                continue

            # Lifecycle for pipeline nodes. Key by run_id (not name) so parallel
            # invocations of the same node — e.g., the fan-out of sub_query
            # branches during report generation — each get their own Step
            # rendered live, instead of colliding on a single open_steps[name].
            if name in NODE_LABEL:
                run_id = ev.get("run_id")

                if kind == "on_chain_start" and run_id and run_id not in open_steps:
                    label = NODE_LABEL[name]
                    # For parallel sub_query branches, surface the section title
                    # in the Step header so the user can see all sections
                    # progressing simultaneously and tell them apart. Without
                    # a unique label, Chainlit's UI consolidates same-named
                    # Steps into one widget and the others go invisible.
                    if name == "sub_query":
                        input_data = (ev.get("data") or {}).get("input") or {}
                        section = (
                            input_data.get("current_section")
                            if isinstance(input_data, dict) else None
                        )
                        # Section may arrive as Pydantic model OR plain dict
                        # depending on serde rehydration timing.
                        if isinstance(section, dict):
                            section_title = section.get("title")
                        elif section is not None:
                            section_title = getattr(section, "title", None)
                        else:
                            section_title = None
                        if section_title:
                            label = f"{label}: {section_title}"
                        else:
                            # Fallback so Chainlit doesn't merge identically-
                            # named parallel Steps: tag with a short run-id.
                            label = f"{label} [{str(run_id)[:8]}]"
                    step = cl.Step(name=label, type="run", default_open=False)
                    icon = NODE_ICON.get(name)
                    if icon:
                        step.icon = icon
                    step.show_input = False
                    # Explicit start timestamp — without it Chainlit's
                    # Using/Used renderer can't tell when the step
                    # actually began and ends up showing stale state.
                    step.start = _utc_now_iso()
                    await step.send()
                    # Kick off a live timer that ticks elapsed time into the
                    # body so the user sees activity instead of an empty
                    # collapsed dropdown. For sub_query branches, the verb
                    # includes the section title so each parallel timer is
                    # distinguishable.
                    started_at = time.perf_counter()
                    timer = asyncio.create_task(_step_timer(step, label, started_at))
                    open_steps[run_id] = {
                        "step": step,
                        "timer": timer,
                        "started_at": started_at,
                        "base_label": label,
                    }

                elif kind == "on_chain_end" and run_id:
                    entry = open_steps.pop(run_id, None)
                    output = ev["data"].get("output") or {}
                    if entry is not None:
                        # Stop the live timer, then overwrite the body with
                        # the real node output + a footer showing how long
                        # this node took.
                        entry["timer"].cancel()
                        try:
                            await entry["timer"]
                        except asyncio.CancelledError:
                            pass
                        elapsed = time.perf_counter() - entry["started_at"]
                        step = entry["step"]
                        # Setting step.end is the signal Chainlit's UI uses
                        # to flip the verb prefix from "Using" to "Used".
                        # The pinned elapsed-time suffix stays as-is.
                        step.end = _utc_now_iso()
                        step.name = f"{entry['base_label']}  ·  {elapsed:.0f}s"
                        step.output = _step_body(name, output)
                        await step.update()
                    if name == "front_agent":
                        fa_intent = output.get("intent")
                        if fa_intent in ("respond", "rechart"):
                            summary_from_state = output.get("summary")
                    if name == "execute_sql" and output.get("rows"):
                        rows_for_chart = output["rows"]
                    if name == "generate_chart" and output.get("chart"):
                        chart_spec = output["chart"]

            # Capture summarize's final state in case streaming missed any tokens.
            # summarize is NOT in NODE_LABEL (its prose streams to the message
            # instead of into a Step), so this elif still gets reached.
            elif name == "summarize" and kind == "on_chain_end":
                output = ev["data"].get("output") or {}
                if output.get("summary"):
                    summary_from_state = output["summary"]

            # Same for generate_response — its streamed text is the visible
            # reply on respond/rechart paths. Defensive fallback in case
            # streaming chunks didn't fire.
            elif name == "generate_response" and kind == "on_chain_end":
                output = ev["data"].get("output") or {}
                if output.get("summary"):
                    summary_from_state = output["summary"]

            # aggregate_report streams the report markdown; capture as fallback.
            elif name == "aggregate_report" and kind == "on_chain_end":
                output = ev["data"].get("output") or {}
                if output.get("summary"):
                    summary_from_state = output["summary"]

    except Exception as e:
        # Cancel any still-ticking timers so they don't keep updating a
        # step that's about to be overwritten with the error label.
        for entry in open_steps.values():
            entry["timer"].cancel()
            try:
                await entry["timer"]
            except asyncio.CancelledError:
                pass
            entry["step"].end = _utc_now_iso()
            entry["step"].output = f"❌ Aborted: {type(e).__name__}"
            await entry["step"].update()
        await cl.Message(content=f"**Graph error:** `{type(e).__name__}: {e}`").send()
        return

    # Respond / rechart paths don't run summarize, so nothing has streamed
    # into final_answer yet. Pipe the front_agent's reply through the smooth
    # streamer so the user sees a typewriter animation just like a data turn,
    # rather than a full reply popping in at once.
    if not final_answer.content and summary_from_state:
        await _stream_smoothed(final_answer, summary_from_state)
    if not final_answer.content:
        final_answer.content = "_(no response produced)_"

    # If we'll need rows for either a chart or the inline-table fallback and
    # didn't see them this turn (rechart path skips execute_sql), pull from
    # checkpointed state. Cheap with MemorySaver.
    if not rows_for_chart and fa_intent in ("data", "rechart"):
        snapshot = await app_graph.aget_state(config)
        rows_for_chart = snapshot.values.get("rows") or []

    # Attach a chart when one was produced. Track whether it actually rendered
    # — kind="none"/"table" or invalid column shapes leave fig=None.
    chart_attached = False
    if chart_spec is not None and rows_for_chart:
        fig = _build_chart(chart_spec, rows_for_chart, context="data turn")
        if fig is not None:
            final_answer.elements = [
                cl.Plotly(name="chart", figure=fig, display="inline")
            ]
            chart_attached = True

    # Report path: attach one Plotly element per section that has a chart.
    # Sections live in checkpointed state; pull the final list once.
    if fa_intent == "report":
        snapshot = await app_graph.aget_state(config)
        report_sections = snapshot.values.get("report_sections") or []
        # Re-sort into outline order
        outline = snapshot.values.get("report_outline") or []
        title_pos = {getattr(s, "title", ""): i for i, s in enumerate(outline)}
        report_sections = sorted(
            report_sections,
            key=lambda s: title_pos.get(getattr(s, "title", ""), len(outline) + 1),
        )
        report_elements = []
        for section in report_sections:
            chart = getattr(section, "chart", None)
            # Prefer rows_for_chart (capped at ~500) — rows_preview is only
            # 10 rows, which makes LLM-generated chart code fail or render
            # empty for trend/top-N plots.
            chart_rows = (
                getattr(section, "rows_for_chart", None)
                or getattr(section, "rows_preview", None)
                or []
            )
            if chart is None or not chart_rows:
                continue
            section_title = getattr(section, "title", "section")
            fig = _build_chart(chart, chart_rows, context=f"section: {section_title}")
            if fig is None:
                continue
            # Section title becomes the element name → Chainlit shows it as a label
            title = getattr(section, "title", "section")
            report_elements.append(cl.Plotly(name=title, figure=fig, display="inline"))
        if report_elements:
            final_answer.elements = (final_answer.elements or []) + report_elements
            chart_attached = True

    # Inline table fallback: data/rechart turns where no chart rendered should
    # still show the rows in the message body, not just buried in the
    # execute_sql Step (which is collapsed by default).
    if not chart_attached and rows_for_chart and fa_intent in ("data", "rechart"):
        body = (final_answer.content or "").rstrip()
        table = _format_rows_table(rows_for_chart)
        final_answer.content = (body + "\n\n**Data**\n\n" + table) if body else ("**Data**\n\n" + table)

    await final_answer.send()
