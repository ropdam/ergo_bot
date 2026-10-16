import json

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolCall

import functions
from sample_docs import DOCS, GREEN_TEA_LINK

ANSWER = f"- bullet with [Green tea]({GREEN_TEA_LINK})"


class ScriptedChatModel(FakeMessagesListChatModel):
    """FakeMessagesListChatModel with a no-op bind_tools for agent tests."""

    def bind_tools(self, tools, **kwargs):
        return self


def _tool_then_answer(query: str = "green tea") -> list[AIMessage]:
    return [
        AIMessage(
            content="",
            tool_calls=[ToolCall(name=functions.RETRIEVER_TOOL_NAME, args={"query": query}, id="call_1")],
        ),
        AIMessage(content=ANSWER),
    ]


@pytest.fixture
def vectorstore() -> FAISS:
    return FAISS.from_documents(DOCS, DeterministicFakeEmbedding(size=32))


@pytest.fixture
def agent(vectorstore):
    tool = functions.build_retriever_tool(vectorstore.as_retriever(search_kwargs={"k": 3}))
    return functions.ErgobotAgent(ScriptedChatModel(responses=_tool_then_answer()), tool)


async def test_ainvoke_returns_answer_and_context(agent):
    result = await agent.ainvoke("Is green tea healthy?")

    assert result["question"] == "Is green tea healthy?"
    assert result["answer"] == ANSWER
    assert len(result["context"]) == 3
    assert all("article_link" in doc.metadata for doc in result["context"])
    assert all(doc.metadata["article_link"].startswith("https://") for doc in result["context"])


async def test_astream_yields_context_before_answer(agent):
    chunks = [chunk async for chunk in agent.astream("Why do athletes eat bananas?")]

    context_idx = next(i for i, c in enumerate(chunks) if "context" in c)
    answer_idxs = [i for i, c in enumerate(chunks) if "answer" in c]
    assert answer_idxs, "no answer chunks streamed"
    assert context_idx < min(answer_idxs)
    assert len(chunks[context_idx]["context"]) == 3
    assert "".join(chunks[i]["answer"] for i in answer_idxs) == ANSWER


async def test_tool_result_includes_urls(vectorstore):
    tool = functions.build_retriever_tool(vectorstore.as_retriever(search_kwargs={"k": 2}))
    msg = await tool.ainvoke(
        {
            "name": functions.RETRIEVER_TOOL_NAME,
            "args": {"query": "green tea"},
            "id": "call_1",
            "type": "tool_call",
        }
    )

    assert GREEN_TEA_LINK in msg.content
    assert "Links (copy exactly):" in msg.content
    assert msg.artifact and all("article_link" in d.metadata for d in msg.artifact)


def test_get_vectorstore_requires_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(functions, "DB_PATH", tmp_path)

    with pytest.raises(RuntimeError, match="make index"):
        functions.get_vectorstore()


def test_get_vectorstore_loads_index_with_manifest_model(monkeypatch, tmp_path, vectorstore):
    vectorstore.save_local(str(tmp_path))
    manifest = {"embedding_model": "text-embedding-3-small", "dimensions": 32, "n_vectors": len(DOCS)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(functions, "DB_PATH", tmp_path)

    db = functions.get_vectorstore()  # constructing OpenAIEmbeddings makes no API call

    assert db.index.ntotal == len(DOCS)
    assert db.embedding_function.model == "text-embedding-3-small"
