"""Embed the scraped articles with OpenAI and write the FAISS index used by the app.

Reads ``data/ergologs.parquet`` (from scrape.py) and replaces ``project/app/db_faiss_OAI``. The new index is built in
a temporary sibling directory and only swapped in once it is completely written, together with a ``manifest.json``.

Usage:
    uv run python project/scraping/build_index.py          # asks for confirmation before calling the API
    uv run python project/scraping/build_index.py --yes    # non-interactive
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("build_index")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "ergologs.parquet"
DEFAULT_OUT = SCRIPT_DIR.parent / "app" / "db_faiss_OAI"
EMBEDDING_MODEL = "text-embedding-3-small"
PRICE_PER_MTOK = 0.02  # USD per 1M input tokens of EMBEDDING_MODEL, for the pre-flight cost estimate only

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 400
METADATA_KEYS = (
    "article_title",
    "article_link",
    "article_date",
    "archive_link",
    "categories",
)


def _str(value: object) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def dataframe_to_documents(df: pd.DataFrame) -> list[Document]:
    """One Document per article; articles longer than CHUNK_SIZE chars are split into chunks (``chunk`` metadata)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs: list[Document] = []
    for row in df.to_dict("records"):
        title = _str(row.get("article_title"))
        text = _str(row.get("article_text"))
        if not text.strip():
            continue
        metadata = {key: _str(row.get(key)) for key in METADATA_KEYS}
        content = f"{title}\n\n{text}"
        if len(content) <= CHUNK_SIZE:
            docs.append(Document(page_content=content, metadata=metadata))
            continue
        for i, chunk in enumerate(splitter.split_text(content)):
            # Later chunks lose the headline; repeat it so every chunk is self-describing for retrieval.
            page_content = chunk if i == 0 or chunk.startswith(title) else f"{title}\n\n{chunk}"
            docs.append(Document(page_content=page_content, metadata={**metadata, "chunk": i}))
    return docs


def estimate_tokens(docs: Sequence[Document]) -> int:
    return sum(len(doc.page_content) for doc in docs) // 4


def build_index(docs: Sequence[Document], embeddings: Embeddings, out_dir: Path, embedding_model: str) -> None:
    """Embed ``docs`` into a FAISS index and replace ``out_dir`` with it.

    The index (+ ``manifest.json``) is written to a temporary sibling directory first and only swapped in once it is
    complete, so a failed run leaves the existing ``out_dir`` untouched.
    """
    tmp_dir = out_dir.with_name(f".{out_dir.name}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    store = FAISS.from_documents(list(docs), embeddings)
    store.save_local(str(tmp_dir))
    manifest = {
        "embedding_model": embedding_model,
        "dimensions": int(store.index.d),
        "n_vectors": int(store.index.ntotal),
        "n_articles": len({doc.metadata.get("article_link") for doc in docs}),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    logger.info("Wrote %d vectors (%d articles) to %s", manifest["n_vectors"], manifest["n_articles"], out_dir)


def confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="scraped parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="FAISS output dir (default: %(default)s)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="skip the confirmation before calling the API",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    if not args.input.exists():
        raise SystemExit(f"{args.input} not found; run scrape.py first")
    df = pd.read_parquet(args.input)
    docs = dataframe_to_documents(df)
    if not docs:
        raise SystemExit(f"No articles with text in {args.input}")

    tokens = estimate_tokens(docs)
    logger.info(
        "%d articles -> %d documents, ~%s tokens with %s (~$%.2f)",
        len(df),
        len(docs),
        f"{tokens:,}",
        EMBEDDING_MODEL,
        tokens / 1e6 * PRICE_PER_MTOK,
    )

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (add it to .env)")
    if not args.yes and not confirm(f"Embed {len(docs)} documents with {EMBEDDING_MODEL} and replace {args.out}?"):
        raise SystemExit("Aborted (pass --yes to run non-interactively)")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, show_progress_bar=True, max_retries=10)
    build_index(docs, embeddings, args.out, embedding_model=EMBEDDING_MODEL)


if __name__ == "__main__":
    main()
