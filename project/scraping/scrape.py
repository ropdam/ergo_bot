"""Crawl ergo-log.com into a parquet file of articles.

Flow: archives.html -> monthly archive pages + index.html -> article refs (deduplicated by link, earliest date wins)
-> article pages -> ``data/ergologs.parquet``. Raw HTML is cached under ``data/html/``: listing pages are always
re-downloaded (so new articles are found), article pages only when not cached yet (``--refresh`` forces it), so an
interrupted crawl can simply be restarted.

Usage:
    uv run python project/scraping/scrape.py                 # full crawl (~4.3k requests at 1 req/s)
    uv run python project/scraping/scrape.py --limit 3 --out project/scraping/data/smoke.parquet
"""

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from urllib3.util.retry import Retry

from ergolog import BASE_URL, Article, ArticleRef, decode, parse_archive_index, parse_article, parse_listing

logger = logging.getLogger("scrape")

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
DEFAULT_CACHE_DIR = DATA_DIR / "html"
DEFAULT_OUT = DATA_DIR / "ergologs.parquet"

USER_AGENT = "ergobot-scraper/2.0 (+https://github.com/ROpdam/ergo_bot)"
TIMEOUT = 20

COLUMNS = ["article_link", "article_title", "article_date", "archive_link", "categories", "article_text", "scraped_at"]


class FetchError(Exception):
    """A page could not be downloaded (HTTP error status, retries exhausted, timeout, ...)."""


def make_session() -> requests.Session:
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = USER_AGENT
    return session


def cache_name(url: str) -> str:
    """Cache file name for a URL: the page's file name (the site is flat), e.g. ``q10fat.html``."""
    name = unquote(Path(urlsplit(url).path).name) or "index.html"
    return name.replace("/", "_").replace("\\", "_")


@dataclass
class Fetcher:
    """Polite HTTP fetcher with an on-disk raw-bytes cache."""

    session: requests.Session
    cache_dir: Path
    delay: float = 1.0
    refresh: bool = False
    _last_request: float = 0.0

    def get_bytes(self, url: str, use_cache: bool = True) -> bytes:
        """Page bytes from the cache, or downloaded (and cached) when missing or ``use_cache`` is False."""
        path = self.cache_dir / cache_name(url)
        if use_cache and path.exists() and not self.refresh:
            return path.read_bytes()

        wait = self._last_request + self.delay - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            response = self.session.get(url, timeout=TIMEOUT)
        except requests.RequestException as exc:
            raise FetchError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self._last_request = time.monotonic()

        if response.status_code != 200:
            raise FetchError(f"HTTP {response.status_code}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return response.content

    def get_text(self, url: str) -> str:
        return decode(self.get_bytes(url))

    def get_fresh_text(self, url: str) -> str:
        """Always re-download (listing pages change); fall back to the cached copy if the download fails."""
        try:
            return decode(self.get_bytes(url, use_cache=False))
        except FetchError as exc:
            path = self.cache_dir / cache_name(url)
            if not path.exists():
                raise
            logger.warning("Using cached copy of %s (%s)", url, exc)
            return decode(path.read_bytes())


def dedupe_refs(refs: list[ArticleRef]) -> list[ArticleRef]:
    """One ref per link, keeping the earliest dated one (undated refs only win when no dated ref exists)."""
    best: dict[str, ArticleRef] = {}
    for ref in refs:
        current = best.get(ref.link)
        if current is None or (ref.date is not None and (current.date is None or ref.date < current.date)):
            best[ref.link] = ref
    return list(best.values())


def article_row(ref: ArticleRef, article: Article, scraped_at: str) -> dict[str, str | None]:
    return {
        "article_link": ref.link,
        "article_title": article.title,
        "article_date": article.date or ref.date,
        "archive_link": ref.listing_url,
        "categories": ";".join(article.categories),
        "article_text": article.text,
        "scraped_at": scraped_at,
    }


def collect_refs(fetcher: Fetcher, failures: list[dict[str, str]]) -> list[ArticleRef]:
    archives_url = BASE_URL + "archives.html"
    try:
        listing_urls = parse_archive_index(fetcher.get_fresh_text(archives_url))
    except FetchError as exc:
        raise SystemExit(f"Could not fetch {archives_url}: {exc}") from exc
    listing_urls.append(BASE_URL + "index.html")  # newest articles are not in the archive pages yet
    logger.info("Found %d listing pages", len(listing_urls))

    refs: list[ArticleRef] = []
    for url in tqdm(listing_urls, desc="listings", unit="page"):
        try:
            page_refs = parse_listing(fetcher.get_fresh_text(url), url)
        except FetchError as exc:
            logger.warning("Skipping listing %s: %s", url, exc)
            failures.append({"url": url, "stage": "listing", "reason": str(exc)})
            continue
        if not page_refs:
            logger.warning("No article links found on %s", url)
        refs.extend(page_refs)

    unique = dedupe_refs(refs)
    logger.info("Found %d article links (%d unique)", len(refs), len(unique))
    return unique


def scrape(fetcher: Fetcher, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    failures: list[dict[str, str]] = []
    refs = collect_refs(fetcher, failures)
    if limit is not None:
        refs = refs[:limit]

    rows: list[dict[str, str | None]] = []
    for ref in tqdm(refs, desc="articles", unit="page"):
        try:
            html = fetcher.get_text(ref.link)
        except FetchError as exc:
            logger.warning("Skipping article %s: %s", ref.link, exc)
            failures.append({"url": ref.link, "stage": "article", "reason": str(exc)})
            continue
        try:
            article = parse_article(html, ref.link)
        except Exception as exc:  # unexpected markup must not abort a long crawl
            logger.exception("Could not parse article %s", ref.link)
            failures.append({"url": ref.link, "stage": "article", "reason": f"parse error: {exc!r}"})
            continue
        if article is None:
            logger.warning("Skipping article %s: no title or empty body", ref.link)
            failures.append({"url": ref.link, "stage": "article", "reason": "unparseable (no title or empty body)"})
            continue
        rows.append(article_row(ref, article, datetime.now(UTC).isoformat(timespec="seconds")))

    articles = pd.DataFrame(rows, columns=COLUMNS)
    failed = pd.DataFrame(failures, columns=["url", "stage", "reason"])
    return articles, failed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output parquet file (default: %(default)s)")
    parser.add_argument("--refresh", action="store_true", help="re-download pages even when cached")
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N articles (smoke runs)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    fetcher = Fetcher(make_session(), DEFAULT_CACHE_DIR, refresh=args.refresh)
    with logging_redirect_tqdm():
        articles, failed = scrape(fetcher, limit=args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(args.out, index=False)
    failed_path = args.out.with_suffix(".failed.csv")
    failed.to_csv(failed_path, index=False)
    logger.info("Wrote %d articles to %s", len(articles), args.out)
    if len(failed):
        logger.warning("%d pages failed, see %s", len(failed), failed_path)


if __name__ == "__main__":
    main()
