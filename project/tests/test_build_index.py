import json
from datetime import datetime

import pandas as pd
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding

import build_index
from build_index import CHUNK_SIZE, METADATA_KEYS, dataframe_to_documents

build = build_index.build_index

DIM = 32


@pytest.fixture
def df() -> pd.DataFrame:
    long_text = "\n\n".join(f"Paragraph {i}. " + "Creatine increases muscle strength. " * 12 for i in range(40))
    assert len(long_text) > CHUNK_SIZE
    return pd.DataFrame(
        [
            {
                "article_link": "https://www.ergo-log.com/q10fat.html",
                "article_title": "Q10 combined with light exercise burns more fat",
                "article_date": "2009-05-31",
                "archive_link": "https://www.ergo-log.com/archives2009.html",
                "categories": "Weight Loss;Q10",
                "article_text": "After taking a capsule containing 30 mg of Q10 your body burns more fat.",
                "scraped_at": "2026-10-11T12:00:00+00:00",
            },
            {
                "article_link": "https://www.ergo-log.com/creatine-long.html",
                "article_title": "A very long creatine article",
                "article_date": None,
                "archive_link": "https://www.ergo-log.com/index.html",
                "categories": "",
                "article_text": long_text,
                "scraped_at": "2026-10-11T12:00:00+00:00",
            },
            {
                "article_link": "https://www.ergo-log.com/empty.html",
                "article_title": "Empty",
                "article_date": "2020-01-01",
                "archive_link": "https://www.ergo-log.com/archives2020a.html",
                "categories": "",
                "article_text": "  ",
                "scraped_at": "2026-10-11T12:00:00+00:00",
            },
        ]
    )


def test_dataframe_to_documents(df):
    docs = dataframe_to_documents(df)
    short = [d for d in docs if d.metadata["article_link"].endswith("q10fat.html")]
    long = [d for d in docs if d.metadata["article_link"].endswith("creatine-long.html")]

    assert len(short) == 1
    assert short[0].page_content.startswith("Q10 combined with light exercise burns more fat\n\nAfter taking")
    assert "chunk" not in short[0].metadata
    assert short[0].metadata["article_date"] == "2009-05-31"
    assert short[0].metadata["categories"] == "Weight Loss;Q10"

    assert len(long) > 1
    assert [d.metadata["chunk"] for d in long] == list(range(len(long)))
    assert all(isinstance(d.metadata["chunk"], int) for d in long)
    assert all(d.page_content.startswith("A very long creatine article") for d in long)
    assert all(len(d.page_content) <= CHUNK_SIZE + 100 for d in long)
    assert long[0].metadata["article_date"] == ""  # None -> "" so the app can always format it

    assert not any(d.metadata["article_link"].endswith("empty.html") for d in docs)
    for doc in docs:
        assert set(METADATA_KEYS) <= set(doc.metadata)


def test_build_index_roundtrip(df, tmp_path):
    docs = dataframe_to_documents(df)
    embeddings = DeterministicFakeEmbedding(size=DIM)
    out_dir = tmp_path / "db_faiss_OAI"

    build(docs, embeddings, out_dir, embedding_model="fake-model")

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["embedding_model"] == "fake-model"
    assert manifest["dimensions"] == DIM
    assert manifest["n_articles"] == 2
    assert manifest["n_vectors"] == len(docs)
    assert manifest["n_vectors"] > manifest["n_articles"]
    assert datetime.fromisoformat(manifest["built_at"]).tzinfo is not None

    store = FAISS.load_local(str(out_dir), embeddings, allow_dangerous_deserialization=True)
    assert store.index.ntotal == len(docs)
    hits = store.similarity_search(docs[0].page_content, k=1)
    assert hits[0].page_content == docs[0].page_content
    for key in (
        "article_title",
        "article_link",
        "article_date",
        "archive_link",
        "categories",
    ):
        assert key in hits[0].metadata

    # no temporary/backup directories left behind
    assert sorted(p.name for p in tmp_path.iterdir()) == ["db_faiss_OAI"]


def test_build_index_replaces_existing_dir(df, tmp_path):
    out_dir = tmp_path / "db"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("old index")
    docs = dataframe_to_documents(df)

    build(docs, DeterministicFakeEmbedding(size=DIM), out_dir, embedding_model="m")

    assert not (out_dir / "stale.txt").exists()
    assert (out_dir / "index.faiss").exists() and (out_dir / "manifest.json").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["db"]


def test_build_index_failure_keeps_old_index(df, tmp_path):
    class Boom(DeterministicFakeEmbedding):
        def embed_documents(self, texts):
            raise RuntimeError("API down")

    out_dir = tmp_path / "db"
    out_dir.mkdir()
    (out_dir / "index.faiss").write_text("old index")

    with pytest.raises(RuntimeError, match="API down"):
        build(dataframe_to_documents(df), Boom(size=DIM), out_dir, embedding_model="m")

    assert (out_dir / "index.faiss").read_text() == "old index"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["db"]


def test_main_requires_confirmation(df, tmp_path, monkeypatch):
    src = tmp_path / "ergologs.parquet"
    df.to_parquet(src, index=False)
    monkeypatch.setattr(build_index, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(build_index, "confirm", lambda prompt: False)

    def fail(*args, **kwargs):
        raise AssertionError("must not embed without confirmation")

    monkeypatch.setattr(build_index, "build_index", fail)
    with pytest.raises(SystemExit, match="Aborted"):
        build_index.main(["--input", str(src), "--out", str(tmp_path / "db")])
    assert not (tmp_path / "db").exists()
