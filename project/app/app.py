"""Chainlit UI for Ergobot."""

import asyncio
import logging

import chainlit as cl
from langchain_core.documents import Document

from functions import format_sources, get_chain

logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "Is ginger healthy? Why?",
    "How to improve endurance?",
    "What are health benefits of green tea and why?",
    "Why do athletes eat bananas?",
    "How to gain muscle mass?",
]


@cl.set_starters
async def set_starters(user: cl.User | None = None, language: str | None = None) -> list[cl.Starter]:
    """Example questions shown on an empty thread."""
    return [cl.Starter(label=question, message=question) for question in EXAMPLE_QUESTIONS]


async def show_sources(docs: list[Document]) -> None:
    """Show the retrieved articles as a collapsible Retriever step."""
    async with cl.Step(name="Retriever", type="retrieval", show_input=False) as step:
        step.output = format_sources(docs) or "No articles found."


@cl.on_message
async def main(message: cl.Message) -> None:
    question = message.content
    try:
        # First call loads the FAISS index from disk; keep it off the event loop.
        chain = await asyncio.to_thread(get_chain)

        answer = cl.Message(content="")
        async for chunk in chain.astream(question):
            if "context" in chunk:
                await show_sources(chunk["context"])
            if token := chunk.get("answer"):
                await answer.stream_token(token)
        await answer.send()
    except Exception:
        logger.exception("Failed to answer question")
        await cl.ErrorMessage(content="Sorry, something went wrong while answering. Please try again.").send()
