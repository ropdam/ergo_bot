"""Agent over the Ergo-Log article archive, with FAISS retrieval as a tool."""

import json
import re
from collections.abc import AsyncIterator
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "db_faiss_OAI"
MODEL = "gpt-6-luna"
K = 5
MANIFEST_NAME = "manifest.json"
RETRIEVER_TOOL_NAME = "search_ergo_log"
_MD_ERGO_LINK = re.compile(r"\[([^\]]+)\]\((https?://(?:www\.)?ergo-log\.com/[^)\s]+)\)")

SYSTEM_PROMPT = """You are Ergobot, an assistant for the Ergo-Log archive \
(https://www.ergo-log.com/), a news site about ergogenic aids studies.

You can call search_ergo_log to look up articles. That is optional: call it when you need \
article evidence; skip it for greetings, chitchat, or when a search is unnecessary.

When you answer from articles:
- Use only what the tool returned.
- Copy markdown links from the tool's Links section exactly — never invent or rewrite URLs.
- Use bullet points; paste one of those links on each point.
- If you don't know, say so. Don't invent answers."""


def get_vectorstore() -> FAISS:
    """Load the FAISS index with the embedding model recorded in its manifest."""
    manifest_file = DB_PATH / MANIFEST_NAME
    if not manifest_file.is_file():
        raise RuntimeError(
            f"No index manifest found at {manifest_file}. Build the index first with `make index` "
            "(or `uv run python project/scraping/build_index.py`)."
        )
    manifest = json.loads(manifest_file.read_text())
    embeddings = OpenAIEmbeddings(model=manifest["embedding_model"])
    # Unpickling is only safe because we build index.pkl ourselves (project/scraping/build_index.py).
    return FAISS.load_local(str(DB_PATH), embeddings, allow_dangerous_deserialization=True)


def format_docs(docs: list[Document]) -> str:
    """Render retrieved docs: a copy-paste Links list, then excerpts."""
    links: list[str] = []
    seen: set[str] = set()
    excerpts: list[str] = []
    for doc in docs:
        title = doc.metadata.get("article_title", "Untitled")
        link = doc.metadata.get("article_link", "")
        when = doc.metadata.get("article_date") or "date unknown"
        if link and link not in seen:
            seen.add(link)
            links.append(f"- [{title}]({link})")
        excerpts.append(f"### {title} ({when})\n{doc.page_content}")
    return "Links (copy exactly):\n" + "\n".join(links) + "\n\n" + "\n\n".join(excerpts)


def rewrite_citations(answer: str, docs: list[Document]) -> str:
    """Fix invented ergo-log URLs by matching link text to retrieved article titles."""
    known_urls = {doc.metadata["article_link"] for doc in docs if doc.metadata.get("article_link")}
    titles = [
        (doc.metadata["article_title"], doc.metadata["article_link"])
        for doc in docs
        if doc.metadata.get("article_link")
    ]

    def repl(match: re.Match[str]) -> str:
        text, url = match.group(1), match.group(2)
        if url in known_urls:
            return match.group(0)
        for title, link in titles:
            if text == title or text in title or title.startswith(text):
                return f"[{title}]({link})"
        return match.group(0)

    return _MD_ERGO_LINK.sub(repl, answer)


def build_retriever_tool(retriever: BaseRetriever) -> StructuredTool:
    """Tool that searches the Ergo-Log index and returns excerpts plus Document artifacts."""

    def _run(query: str) -> tuple[str, list[Document]]:
        docs = retriever.invoke(query)
        return format_docs(docs), docs

    async def _arun(query: str) -> tuple[str, list[Document]]:
        docs = await retriever.ainvoke(query)
        return format_docs(docs), docs

    return StructuredTool.from_function(
        name=RETRIEVER_TOOL_NAME,
        description=(
            "Optional: search the Ergo-Log article archive when you need study evidence. "
            "Returns article excerpts with titles, URLs, and dates. Do not call for greetings."
        ),
        func=_run,
        coroutine=_arun,
        response_format="content_and_artifact",
    )


class ErgobotAgent:
    """Tool-calling agent: question str -> streamed {"context"} / {"answer"} chunks."""

    def __init__(self, llm: BaseChatModel, tool: StructuredTool):
        # tool_choice="auto": model may answer without calling the retriever.
        self._llm = llm.bind_tools([tool], tool_choice="auto")
        self._tool = tool

    async def ainvoke(self, question: str) -> dict:
        context: list[Document] = []
        answer_parts: list[str] = []
        async for chunk in self.astream(question):
            if docs := chunk.get("context"):
                context.extend(docs)
            if token := chunk.get("answer"):
                answer_parts.append(token)
        return {"question": question, "context": context, "answer": "".join(answer_parts)}

    async def astream(self, question: str) -> AsyncIterator[dict]:
        """Yield {"context": docs} when the retriever tool runs, then {"answer": token} chunks."""
        messages: list = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
        retrieved: list[Document] = []

        while True:
            gathered: AIMessage | None = None
            content_chunks: list[str] = []
            async for chunk in self._llm.astream(messages):
                gathered = chunk if gathered is None else gathered + chunk
                # chunk.content may be a list of blocks; .text normalizes to text (str subclass).
                if text := str(chunk.text):
                    content_chunks.append(text)

            assert gathered is not None
            if gathered.tool_calls:
                messages.append(gathered)
                for call in gathered.tool_calls:
                    tool_msg: ToolMessage = await self._tool.ainvoke(
                        {
                            "name": call["name"],
                            "args": call["args"],
                            "id": call["id"],
                            "type": "tool_call",
                        }
                    )
                    if call["name"] == RETRIEVER_TOOL_NAME and tool_msg.artifact:
                        retrieved.extend(tool_msg.artifact)
                        yield {"context": tool_msg.artifact}
                    messages.append(tool_msg)
                continue

            answer = rewrite_citations("".join(content_chunks), retrieved) if retrieved else "".join(content_chunks)
            if answer:
                yield {"answer": answer}
            break


@lru_cache(maxsize=1)
def get_agent() -> ErgobotAgent:
    """The app's agent, built once per process and shared across sessions."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": K})
    return ErgobotAgent(ChatOpenAI(model=MODEL), build_retriever_tool(retriever))


def format_sources(docs: list[Document]) -> str:
    """Numbered markdown list of unique source articles, in retrieval order."""
    seen: set[str] = set()
    lines: list[str] = []
    for doc in docs:
        meta = doc.metadata
        link = meta["article_link"]
        if link in seen:
            continue
        seen.add(link)
        entry = f"[{meta['article_title']}]({link})"
        if article_date := meta.get("article_date"):
            entry += f" — {date.fromisoformat(article_date):%d-%m-%Y}"
        lines.append(f"{len(lines) + 1}. {entry}")
    return "\n".join(lines)
