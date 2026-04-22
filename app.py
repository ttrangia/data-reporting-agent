import os
from dotenv import load_dotenv
load_dotenv()  # must come before any imports that read env vars

import chainlit as cl
from langchain_core.messages import HumanMessage

from agent.graph import app_graph
from agent.state import initial_state


@cl.on_chat_start
async def on_chat_start():
    """Called when a new chat session begins."""
    cl.user_session.set("thread_id", cl.context.session.id)
    await cl.Message(
        content="Ready. Ask me anything about the sales database."
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    # Build initial state for this turn
    state = initial_state(message.content)
    state["messages"] = [HumanMessage(content=message.content)]

    # Placeholder message we'll update as the graph streams
    response = cl.Message(content="")
    await response.send()

    # Invoke the graph — for v0, just wait for final state
    final_state = await app_graph.ainvoke(state, config=config)

    response.content = final_state.get("summary") or "No response."
    await response.update()