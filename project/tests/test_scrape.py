"""Offline tests for the crawler: a fake session serves the saved fixture pages."""

from pathlib import Path

import pandas as pd
import pytest
import requests

import scrape
from ergolog import BASE_URL

FIXTURES = Path(__file__).parent / "fixtures" / "html"
ARTICLES = ("chitosan-microplastics.html", "q10fat.html")


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


class FakeSession:
    """Serves fixture files by page name; everything else is a 404. Records requested URLs."""

    def __init__(self, pages: dict[str, bytes], down: bool = False):
        self.pages = pages
        self.down = down
        self.requested: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.requested.append(url)
        if self.down:
            raise requests.ConnectionError("site is down")
        name = scrape.cache_name(url)
        return FakeResponse(200, self.pages[name]) if name in self.pages else FakeResponse(404)


def fixture_pages() -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in FIXTURES.glob("*.html")}


@pytest.fixture
def fetcher(tmp_path):
    return scrape.Fetcher(FakeSession(fixture_pages()), tmp_path / "html", delay=0)


def test_article_pages_are_served_from_cache(fetcher):
    url = BASE_URL + "q10fat.html"
    first = fetcher.get_text(url)
    second = fetcher.get_text(url)
    assert first == second
    assert fetcher.session.requested == [url]
    assert (fetcher.cache_dir / "q10fat.html").read_bytes() == (FIXTURES / "q10fat.html").read_bytes()


def test_refresh_redownloads_cached_pages(fetcher):
    url = BASE_URL + "q10fat.html"
    fetcher.get_text(url)
    fetcher.refresh = True
    fetcher.get_text(url)
    assert fetcher.session.requested == [url, url]


def test_listing_pages_are_always_redownloaded(fetcher):
    url = BASE_URL + "archives.html"
    fetcher.get_fresh_text(url)
    fetcher.get_fresh_text(url)
    assert fetcher.session.requested == [url, url]


def test_listing_falls_back_to_cache_when_download_fails(fetcher):
    url = BASE_URL + "archives.html"
    cached = fetcher.get_fresh_text(url)
    fetcher.session.down = True
    assert fetcher.get_fresh_text(url) == cached


def test_fetch_errors_without_cache_are_raised(fetcher):
    with pytest.raises(scrape.FetchError, match="HTTP 404"):
        fetcher.get_text(BASE_URL + "does-not-exist.html")
    fetcher.session.down = True
    with pytest.raises(scrape.FetchError, match="ConnectionError"):
        fetcher.get_fresh_text(BASE_URL + "archives.html")


def test_dedupe_refs_keeps_earliest_date():
    refs = [
        scrape.ArticleRef("a.html", "2017-09-18", "archives2017b.html"),
        scrape.ArticleRef("a.html", "2009-05-31", "archives2009.html"),
        scrape.ArticleRef("a.html", None, "index.html"),
        scrape.ArticleRef("b.html", None, "index.html"),
    ]
    unique = {ref.link: ref for ref in scrape.dedupe_refs(refs)}
    assert unique["a.html"].date == "2009-05-31"
    assert unique["a.html"].listing_url == "archives2009.html"
    assert unique["b.html"].date is None


def test_article_row_prefers_article_date_over_listing_date():
    ref = scrape.ArticleRef(BASE_URL + "q10fat.html", "2017-09-18", BASE_URL + "archives2017b.html")
    article = scrape.Article(ref.link, "Q10", "2009-05-31", "Text.")
    assert scrape.article_row(ref, article, "now")["article_date"] == "2009-05-31"
    article.date = None  # no date on the article page: fall back to the listing date
    assert scrape.article_row(ref, article, "now")["article_date"] == "2017-09-18"


def crawl(fetcher, monkeypatch, broken: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crawl the fixtures, with the article refs limited to the two saved articles."""
    collect = scrape.collect_refs

    def only_fixture_articles(fetcher, failures):
        return [ref for ref in collect(fetcher, failures) if scrape.cache_name(ref.link) in ARTICLES]

    monkeypatch.setattr(scrape, "collect_refs", only_fixture_articles)
    if broken:
        parse = scrape.parse_article

        def parse_or_crash(html, url):
            if url.endswith(broken):
                raise AttributeError("unexpected markup")
            return parse(html, url)

        monkeypatch.setattr(scrape, "parse_article", parse_or_crash)
    return scrape.scrape(fetcher)


def test_scrape_builds_article_rows(fetcher, monkeypatch):
    articles, failed = crawl(fetcher, monkeypatch)

    assert list(articles.columns) == scrape.COLUMNS
    assert sorted(articles["article_link"]) == sorted(BASE_URL + name for name in ARTICLES)
    q10 = articles.set_index("article_link").loc[BASE_URL + "q10fat.html"]
    assert q10["archive_link"].startswith(BASE_URL + "archives")
    assert q10["article_title"] and "More:" not in q10["article_text"]
    # Listing pages that are not saved as fixtures 404 and are reported, not fatal
    assert set(failed["stage"]) == {"listing"}


def test_scrape_survives_parser_crash(fetcher, monkeypatch):
    articles, failed = crawl(fetcher, monkeypatch, broken="q10fat.html")

    assert list(articles["article_link"]) == [BASE_URL + "chitosan-microplastics.html"]
    crash = failed[failed["stage"] == "article"]
    assert list(crash["url"]) == [BASE_URL + "q10fat.html"]
    assert "unexpected markup" in crash["reason"].iloc[0]


def test_main_writes_parquet_and_failed_csv(tmp_path, monkeypatch):
    session = FakeSession(fixture_pages())
    monkeypatch.setattr(scrape, "make_session", lambda: session)
    monkeypatch.setattr(scrape, "DEFAULT_CACHE_DIR", tmp_path / "html")
    monkeypatch.setattr(
        scrape,
        "collect_refs",
        lambda fetcher, failures: [scrape.ArticleRef(BASE_URL + "q10fat.html", "2009-05-31", "x")],
    )
    out = tmp_path / "out" / "articles.parquet"

    scrape.main(["--out", str(out)])

    df = pd.read_parquet(out)
    assert len(df) == 1 and df["article_link"].iloc[0] == BASE_URL + "q10fat.html"
    assert (out.parent / "articles.failed.csv").is_file()
