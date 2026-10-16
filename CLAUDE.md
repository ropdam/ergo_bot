# Ergobot

RAG chatbot over the [Ergo-Log](https://www.ergo-log.com/) archive: Chainlit UI (`project/app/`), tool-calling agent
with FAISS retriever tool + OpenAI `gpt-6-luna`, scraper/indexer in `project/scraping/`, deployed on Cloud Run.
Follow the `karpathy-guidelines` skill: simplest code that works, surgical changes, verify with tests.

## Rules
- uv project at the repo root: `uv run ...` / Makefile only, add deps with `uv add [--group scrape|--dev]`. Never create `requirements*.txt`.
- `make lint test` before finishing. Use `make scrape-smoke` + fixture tests; never run `make scrape` (~75 min against a third-party site) or `make index` (costs OpenAI money) unprompted.
- `project/app/` and `project/scraping/` are script dirs, not packages: import by bare module name (`from functions import ...`).

## Non-obvious contracts
- Index (`project/app/db_faiss_OAI/`, committed) must have `manifest.json`; the app embeds queries with its `embedding_model` and refuses an index without one (prevents silent ada-002 vs 3-small mismatch, both 1536-dim). `index.pkl` is a pickle: only load indexes we built.
- Document metadata used by the app: `article_title`, `article_link`, `article_date` (ISO string). One doc per article; only articles >6000 chars are split (`chunk` key). Empty `categories` on pre-~2015 pages is expected.
- Retriever is a tool (`search_ergo_log`); tool output includes `URL: {article_link}` so the agent can cite real links.
- When streamed, the agent emits `context` (tool artifacts) before answer tokens; `app.py` relies on this to show sources first.
- Model, k and embedding model are constants in code, not env vars (deliberately: gpt-6-luna streams fine with default params). Only `OPENAI_API_KEY` comes from the environment.

## Scraping ergo-log.com
- Hand-written HTML without charset: old pages windows-1252, new UTF-8. Decode raw bytes with `ergolog.decode`, never `response.text`.
- lxml closes `<p>` at inner `<table>`s, so `parse_article` walks the content cell from `p#kopindex` to the first `p#bodytekst3` instead of selecting `p#bodytekst`.
- Listing pages (`archives*.html`, `index.html`) are always re-downloaded; article pages are cached in `project/scraping/data/html/`. Bad pages go to `<output>.failed.csv`, never abort the crawl.
- If the markup changes: save new pages to `project/tests/fixtures/html/`, fix `ergolog.py`, keep tests green.

## Deployment
- Push to `master` deploys (`deploy.yml` → reuses `ci.yml` → Cloud Run `ergobot-image`, europe-west4). Flow: feature branch → `develop` → `master`.
- Docker build context is the repo root; `.dockerignore` is an allowlist. `OPENAI_API_KEY` lives on the Cloud Run service, not in GitHub.
- CI's docker smoke test asserts the shipped index loads, so it fails until the index is rebuilt with a manifest.

## Gotchas
- `langchain-community` (FAISS) is being sunset; `langchain-faiss` on PyPI is unofficial, don't switch to it.
- Chainlit 2.x: GitHub link only via `[[UI.header_links]]`; runtime writes `.chainlit/translations/` and `.files/` (gitignored) into its cwd, so in Docker the app dir must be owned by the non-root `app` user.
- Old notebook (unmerged `feature/scraping` branch) is superseded; its bugs (missed archive pages, "More:" leakage, cp1252 junk) are covered by tests.
