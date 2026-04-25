import asyncio
import os
from dotenv import load_dotenv
load_dotenv()  # must come before any imports that read env vars

import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.db import warmup
from agent.graph import app_graph
from agent.state import turn_input

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


# Upstream pipeline nodes — each gets its own cl.Step. summarize is excluded
# because its prose streams directly into the user-facing cl.Message instead
# of into a Step body (no point in duplicating the answer).
NODE_LABEL = {
    "front_agent":  "Understanding the question",
    "generate_sql": "Generating SQL",
    "validate_sql": "Validating query",
    "execute_sql":  "Executing on Postgres",
}

NODE_ICON = {
    "front_agent":  "brain",
    "generate_sql": "code-2",
    "validate_sql": "shield-check",
    "execute_sql":  "database",
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
    return ""


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
                    if name == "front_agent" and output.get("intent") == "respond":
                        summary_from_state = output.get("summary")

            # Capture summarize's final state in case streaming missed any tokens
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

    await final_answer.send()
