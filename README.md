# Ergobot

A chatbot that answers questions using the archived articles of [Ergo-Log](https://www.ergo-log.com/).
Questions are matched against a FAISS index of all scraped articles; the best matches are passed to an
OpenAI model (`gpt-6-luna`, set in `project/app/functions.py`) that answers in bullet points and links the
source articles.

- UI: [Chainlit](https://chainlit.io/) (`project/app/app.py`)
- RAG chain: LangChain (LCEL) with `langchain-openai` + FAISS (`project/app/functions.py`)
- Data pipeline: `project/scraping/` (scrape ergo-log.com → parquet → FAISS index)
- Deployment: Docker image on Google Cloud Run, via GitHub Actions on push to `master`

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13 is installed by uv automatically).

```bash
uv sync
```

```bash
cp .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`.

Dependencies live in [pyproject.toml](pyproject.toml) and are locked in `uv.lock`: runtime deps are the
project dependencies (the only ones installed in the Docker image); the `scrape` and `dev` groups are installed
locally by default. Add packages with `uv add <pkg>` or `uv add --group scrape <pkg>`.

## Refreshing the article index

The index in `project/app/db_faiss_OAI/` is built in two steps.

**1. Scrape ergo-log.com** (about 4,300 pages at 1 request/second, so roughly 75 minutes):

```bash
make scrape
```

- Writes `project/scraping/data/ergologs.parquet` and `ergologs.failed.csv` with pages that could not be fetched or
  parsed.
- Raw HTML is cached in `project/scraping/data/html/`. The ~37 listing pages are always re-downloaded so new articles
  are found; article pages are only fetched when they are not cached yet (pass `--refresh` to `scrape.py` to
  re-download everything). An interrupted scrape can simply be restarted.
- Pages in the failures file that could not be downloaded are retried on the next run; to retry one that could not be
  parsed, delete its file in `project/scraping/data/html/` and re-run.
- `make scrape-smoke` scrapes just a few articles to check that everything works.

**2. Build the FAISS index** with OpenAI embeddings (`text-embedding-3-small`, roughly $0.06 for the full archive):

```bash
make index
```

- Prints a cost estimate and asks for confirmation before calling the OpenAI API.
- Replaces `project/app/db_faiss_OAI/` only after the new index is fully written, and records the embedding model
  in `manifest.json`. The app reads that manifest so query embeddings always match the index.
- The index is a pickle that the app loads with `allow_dangerous_deserialization=True`. That is only safe because
  we build it ourselves: never load an index from an untrusted source.

Commit the new `project/app/db_faiss_OAI/` and deploy to ship the refreshed data.

## Running locally

```bash
make run
```

Opens the Chainlit app on http://localhost:8000. Or run it in Docker the way Cloud Run does:

```bash
make docker-build docker-run
```

## Development

```bash
make lint test
```

- Ruff handles linting and formatting (`make format`), pytest runs the tests in `project/tests/`.
- Install the git hooks once with `uv run pre-commit install`.
- The parser tests run against saved pages in `project/tests/fixtures/html/`; if ergo-log.com changes its markup,
  save fresh copies there and update `project/scraping/ergolog.py`.

## Deployment

- `.github/workflows/ci.yml` runs lint, tests and a Docker smoke test (which also checks that the shipped index loads)
  on pull requests and pushes to `develop`.
- `.github/workflows/deploy.yml` runs CI, builds `project/Dockerfile.prod` and deploys it to the Cloud Run service
  `ergobot-image` (europe-west4) on every push to `master`. It needs the `GCP_PROJECT_ID` and `GCP_CREDENTIALS`
  repository secrets. `OPENAI_API_KEY` is configured on the Cloud Run service itself.

## Notes

- `langchain-community` (which provides the FAISS vector store) is being sunset by LangChain. It still works; when an
  official standalone FAISS integration appears, swap the import in `functions.py` and `build_index.py`.
