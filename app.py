import os
from dotenv import load_dotenv
load_dotenv()  # must come before any imports that read env vars

import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.db import warmup
from agent.graph import app_graph
from agent.state import turn_input

# Wake Neon and prime the schema cache before the first user message.
warmup()


PREVIEW_ROWS = 10

# Friendly labels shown in the cl.Step header for each graph node.
NODE_LABELS = {
    "front_agent":  "Understanding your question",
    "generate_sql": "Generating SQL",
    "validate_sql": "Validating query",
    "execute_sql":  "Executing on Postgres",
    "summarize":    "Writing summary",
}


def _format_rows_preview(rows: list[dict] | None) -> str:
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


def _format_node_output(node: str, output: dict) -> str:
    """Render a node's state-update as a human-readable Step body."""
    if node == "front_agent":
        if output.get("intent") == "data":
            dq = output.get("data_question") or "(none)"
            return f"**Routed to data pipeline.**\n\nRefined question:\n> {dq}"
        return "**Direct response — no SQL needed.**"
    if node == "generate_sql":
        sql = output.get("sql") or "(no SQL produced)"
        return f"```sql\n{sql}\n```"
    if node == "validate_sql":
        if output.get("sql_error"):
            return f"❌ {output['sql_error']}"
        return "✓ Passed"
    if node == "execute_sql":
        if output.get("sql_error"):
            return f"❌ {output['sql_error']}"
        rows = output.get("rows") or []
        body = f"✓ Returned {len(rows)} row(s)"
        if rows:
            body += f"\n\n{_format_rows_preview(rows)}"
        return body
    if node == "summarize":
        return "✓ Summary written"
    return "(no output)"


def _extract_text(chunk_content) -> str:
    """AIMessageChunk.content can be str or a list of content blocks (Anthropic)."""
    if isinstance(chunk_content, str):
        return chunk_content
    if isinstance(chunk_content, list):
        return "".join(
            b.get("text", "")
            for b in chunk_content
            if isinstance(b, dict) and b.get("type") == "text"
        )
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
    summary_msg: cl.Message | None = None
    final_summary_from_state: str | None = None  # respond-path summary (no streaming)

    try:
        async for ev in app_graph.astream_events(
            turn_input(message.content, HumanMessage(content=message.content)),
            config=config,
            version="v2",
        ):
            kind = ev["event"]
            name = ev.get("name")
            meta = ev.get("metadata") or {}
            node = meta.get("langgraph_node")

            # Node lifecycle: open a Step on entry, close it on exit.
            # We filter on `name in NODE_LABELS` because LangGraph emits
            # on_chain_start/end for the wrapping graph and other internals too.
            if kind == "on_chain_start" and name in NODE_LABELS and name not in open_steps:
                step = cl.Step(name=NODE_LABELS[name], type="tool")
                await step.send()
                open_steps[name] = step

            elif kind == "on_chain_end" and name in NODE_LABELS:
                step = open_steps.pop(name, None)
                output = ev["data"].get("output") or {}
                if step is not None:
                    step.output = _format_node_output(name, output)
                    await step.update()
                # Capture the front_agent's response_text for respond-paths
                # (these don't go through summarize so token streaming doesn't apply)
                if name == "front_agent" and output.get("intent") == "respond":
                    final_summary_from_state = output.get("summary")

            # Stream summary tokens token-by-token to the user-facing message.
            elif kind == "on_chat_model_stream" and node == "summarize":
                if summary_msg is None:
                    summary_msg = cl.Message(content="")
                    await summary_msg.send()
                chunk = ev["data"].get("chunk")
                if chunk is not None:
                    text = _extract_text(chunk.content)
                    if text:
                        await summary_msg.stream_token(text)

    except Exception as e:
        # Close any open steps before surfacing the error
        for step in open_steps.values():
            step.output = f"❌ Aborted: {type(e).__name__}"
            await step.update()
        await cl.Message(content=f"**Graph error:** `{type(e).__name__}: {e}`").send()
        return

    if summary_msg is not None:
        # Finalize the streamed message — no-op content change, just commits the buffer
        await summary_msg.update()
    elif final_summary_from_state:
        # Respond-path (no summarize node ran); send the front agent's reply as a message
        await cl.Message(content=final_summary_from_state).send()
