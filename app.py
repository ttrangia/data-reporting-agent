import asyncio
import os
from dotenv import load_dotenv
load_dotenv()  # must come before any imports that read env vars

import chainlit as cl
import pandas as pd
import plotly.express as px
from langchain_core.messages import HumanMessage

from agent.db import warmup
from agent.graph import app_graph
from agent.state import ChartSpec, turn_input

warmup()


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
    "front_agent":    "Understanding the question",
    "generate_sql":   "Generating SQL",
    "validate_sql":   "Validating query",
    "execute_sql":    "Executing on Postgres",
    "diagnose_empty": "Investigating empty result",
    "generate_chart": "Designing chart",
}

NODE_ICON = {
    "front_agent":    "brain",
    "generate_sql":   "code-2",
    "validate_sql":   "shield-check",
    "execute_sql":    "database",
    "diagnose_empty": "search-check",
    "generate_chart": "chart-bar",
}


def _step_body(node: str, output: dict) -> str:
    if node == "front_agent":
        if output.get("intent") == "data":
            dq = output.get("data_question") or "(none)"
            return f"_Refined as:_\n\n> {dq}"
        return "_Direct reply — no SQL needed._"
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
    # Fallback for any node added to NODE_LABEL without a custom branch above.
    return _generic_step_body(output)


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
    lines = [f"**{chart.kind.title()} chart**"]
    if chart.title:
        lines.append(f"_{chart.title}_")
    if chart.x and chart.y:
        lines.append(f"x = `{chart.x}`  ·  y = `{chart.y}`")
    if chart.reasoning:
        lines.append(chart.reasoning)
    return "\n\n".join(lines)


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


def _build_chart(spec: ChartSpec, rows: list[dict]):
    """Build a Plotly figure from a ChartSpec + rows. Returns None if the spec
    is unrenderable (kind="none"/"table", missing rows, missing columns).

    Honors layout knobs on bar/line: barmode, orientation, facet_col, sort_by.
    Silently skips a knob if it's malformed (e.g., facet_col not in df.columns)
    rather than failing the whole chart — the 2D base chart is still useful.
    """
    if not rows or spec.kind in (None, "none", "table"):
        return None
    df = pd.DataFrame(rows)

    # Sort the dataframe if a sort column was specified.
    if spec.sort_by and spec.sort_by in df.columns:
        df = df.sort_values(spec.sort_by, ascending=not bool(spec.sort_desc))

    title = spec.title or ""
    color = spec.group if spec.kind in ("bar", "line") and spec.group in df.columns else None
    facet_col = spec.facet_col if spec.kind in ("bar", "line") and spec.facet_col in df.columns else None
    orientation = spec.orientation or "v"

    # Default barmode: "group" if a color dimension exists, otherwise relative.
    barmode = spec.barmode or ("group" if color else "relative")

    try:
        if spec.kind == "bar":
            return px.bar(
                df, x=spec.x, y=spec.y,
                color=color, facet_col=facet_col,
                title=title, barmode=barmode, orientation=orientation,
            )
        if spec.kind == "line":
            return px.line(
                df, x=spec.x, y=spec.y,
                color=color, facet_col=facet_col,
                title=title, markers=True, orientation=orientation,
            )
        if spec.kind == "pie":
            return px.pie(df, names=spec.x, values=spec.y, title=title)
    except Exception:
        # Bad column types, missing data, etc. — silently skip the chart;
        # the prose summary is still the primary answer.
        return None
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
    config = {"configurable": {"thread_id": thread_id}}

    open_steps: dict[str, cl.Step] = {}
    final_answer = cl.Message(content="")
    summary_from_state: str | None = None
    rows_for_chart: list[dict] = []
    chart_spec: ChartSpec | None = None
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

            # Stream summarize LLM tokens straight into the user-facing message.
            # No Step for summarize — the streaming message IS its visible output.
            if kind == "on_chat_model_stream" and node_for_metadata == "summarize":
                chunk = ev["data"].get("chunk")
                if chunk is not None:
                    text = _extract_text(chunk.content)
                    if text:
                        await _stream_smoothed(final_answer, text)
                continue

            # Lifecycle for the four upstream pipeline nodes.
            if name in NODE_LABEL:
                if kind == "on_chain_start" and name not in open_steps:
                    step = cl.Step(
                        name=NODE_LABEL[name],
                        type="run",
                        default_open=False,
                    )
                    icon = NODE_ICON.get(name)
                    if icon:
                        step.icon = icon
                    step.show_input = False
                    await step.send()
                    open_steps[name] = step

                elif kind == "on_chain_end":
                    step = open_steps.pop(name, None)
                    output = ev["data"].get("output") or {}
                    if step is not None:
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

    except Exception as e:
        for step in open_steps.values():
            step.output = f"❌ Aborted: {type(e).__name__}"
            await step.update()
        await cl.Message(content=f"**Graph error:** `{type(e).__name__}: {e}`").send()
        return

    if not final_answer.content and summary_from_state:
        final_answer.content = summary_from_state
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
        fig = _build_chart(chart_spec, rows_for_chart)
        if fig is not None:
            final_answer.elements = [
                cl.Plotly(name="chart", figure=fig, display="inline")
            ]
            chart_attached = True

    # Inline table fallback: data/rechart turns where no chart rendered should
    # still show the rows in the message body, not just buried in the
    # execute_sql Step (which is collapsed by default).
    if not chart_attached and rows_for_chart and fa_intent in ("data", "rechart"):
        body = (final_answer.content or "").rstrip()
        table = _format_rows_table(rows_for_chart)
        final_answer.content = (body + "\n\n**Data**\n\n" + table) if body else ("**Data**\n\n" + table)

    await final_answer.send()
