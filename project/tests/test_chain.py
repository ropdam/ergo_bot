import json

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

import functions
from sample_docs import DOCS

ANSWER = "- bullet"


@pytest.fixture
def vectorstore() -> FAISS:
    return FAISS.from_documents(DOCS, DeterministicFakeEmbedding(size=32))


@pytest.fixture
def chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return functions.build_chain(retriever, FakeListChatModel(responses=[ANSWER]))


async def test_ainvoke_returns_answer_and_context(chain):
    result = await chain.ainvoke("Is green tea healthy?")

    assert result["question"] == "Is green tea healthy?"
    assert result["answer"] == ANSWER
    assert len(result["context"]) == 3
    assert all("article_link" in doc.metadata for doc in result["context"])


async def test_astream_yields_context_before_answer(chain):
    chunks = [chunk async for chunk in chain.astream("Why do athletes eat bananas?")]

    context_idx = next(i for i, c in enumerate(chunks) if "context" in c)
    answer_idxs = [i for i, c in enumerate(chunks) if "answer" in c]
    assert answer_idxs, "no answer chunks streamed"
    assert context_idx < min(answer_idxs)
    assert len(chunks[context_idx]["context"]) == 3
    assert "".join(chunks[i]["answer"] for i in answer_idxs) == ANSWER


async def test_prompt_contains_context_and_question(vectorstore):
    captured = {}

    class RecordingChatModel(FakeListChatModel):
        async def _agenerate(self, messages, *args, **kwargs):
            captured["messages"] = messages
            return await super()._agenerate(messages, *args, **kwargs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    chain = functions.build_chain(retriever, RecordingChatModel(responses=[ANSWER]))
    result = await chain.ainvoke("How to gain muscle mass?")

    system, human = captured["messages"]
    assert human.content == "How to gain muscle mass?"
    for doc in result["context"]:
        assert doc.metadata["article_title"] in system.content
    assert "don't know" in system.content


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
