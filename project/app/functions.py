"""Retrieval-augmented QA chain over the Ergo-Log article archive."""

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "db_faiss_OAI"
MODEL = "gpt-6-luna"
K = 5
MANIFEST_NAME = "manifest.json"

SYSTEM_PROMPT = """You are Ergobot, an assistant that answers questions using articles from the \
Ergo-Log archive (https://www.ergo-log.com/), a news site about ergogenic aids studies.

Rules:
- Answer only using the Ergo-Log articles provided below.
- Answer using bullet points, include a link to the article for each point.
- If the articles do not contain the answer, say that you don't know. Don't try to make up an answer.

Ergo-Log articles:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


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
    """Render retrieved documents as prompt context."""
    return "\n\n".join(
        f"### {doc.metadata.get('article_title', 'Untitled')} ({doc.metadata.get('article_date') or 'date unknown'})\n"
        f"{doc.page_content}"
        for doc in docs
    )


def build_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    """LCEL chain: question str -> {"question", "context", "answer"}.

    When streamed, the `context` chunk is emitted before any `answer` token chunks.
    """
    answer = (
        RunnableLambda(lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(context=retriever, question=RunnablePassthrough()).assign(answer=answer)


@lru_cache(maxsize=1)
def get_chain() -> Runnable:
    """The app's chain, built once per process and shared across sessions."""
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": K})
    return build_chain(retriever, ChatOpenAI(model=MODEL))


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
