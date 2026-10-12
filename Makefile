.PHONY: sync lint format test scrape scrape-smoke index run docker-build docker-run

IMAGE ?= ergobot

sync:
	uv sync

lint:
	uv run ruff check
	uv run ruff format --check

format:
	uv run ruff check --fix
	uv run ruff format

test:
	uv run pytest

# Full crawl of ergo-log.com (~4.3k pages at 1 req/s, roughly 75 minutes). Raw HTML is cached.
scrape:
	uv run python project/scraping/scrape.py

# Quick end-to-end check of the scraper on a few articles
scrape-smoke:
	uv run python project/scraping/scrape.py --limit 5 --out project/scraping/data/smoke.parquet

# Embed the scraped articles with OpenAI and replace project/app/db_faiss_OAI
index:
	uv run python project/scraping/build_index.py

run:
	cd project/app && uv run chainlit run app.py -w

docker-build:
	docker build -f project/Dockerfile.prod -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env $(IMAGE)
