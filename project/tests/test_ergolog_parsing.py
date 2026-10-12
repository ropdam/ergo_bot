import re
from pathlib import Path

import pytest

import ergolog
from ergolog import BASE_URL, decode, parse_archive_index, parse_article, parse_listing

FIXTURES = Path(__file__).parent / "fixtures" / "html"


def load(name: str) -> str:
    return decode((FIXTURES / name).read_bytes())


# --- decoding ---------------------------------------------------------------------------------------------------


def test_decode_windows_1252_page():
    raw = (FIXTURES / "q10fat.html").read_bytes()
    assert b"\x96" in raw  # old pages are windows-1252 (served without charset)
    text = decode(raw)
    assert "\x96" not in text
    assert "control group – which" in text


def test_decode_utf8_page():
    assert "�" not in load("chitosan-microplastics.html")


def test_parse_date():
    assert ergolog.parse_date("bla 31.05.2009 bla") == "2009-05-31"
    assert ergolog.parse_date("(01-08-2026)") == "2026-08-01"
    assert ergolog.parse_date("00.13.2020") is None
    assert ergolog.parse_date(None) is None


# --- archive index ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def archive_index() -> list[str]:
    return parse_archive_index(load("archives.html"))


def test_archive_index_covers_all_years(archive_index):
    names = [url.removeprefix(BASE_URL) for url in archive_index]
    assert "archives2008.html" in names
    for year in range(2009, 2013):  # first halves without the a/b suffix (missed by the old notebook)
        assert f"archives{year}.html" in names
        assert f"archives{year}b.html" in names
    assert "archives2026.html" in names
    assert "archives2025a.html" in names and "archives2025b.html" in names
    assert len(names) == 36


def test_archive_index_has_no_fragments_or_duplicates(archive_index):
    assert all("#" not in url for url in archive_index)
    assert len(archive_index) == len(set(archive_index))
    assert all(url.startswith(BASE_URL) for url in archive_index)


# --- listings ---------------------------------------------------------------------------------------------------


def refs_by_slug(name: str) -> dict[str, ergolog.ArticleRef]:
    refs = parse_listing(load(name), BASE_URL + name)
    return {ref.link.removeprefix(BASE_URL): ref for ref in refs}


def test_listing_2026():
    refs = refs_by_slug("archives2026.html")
    assert len(refs) == 45
    assert refs["chitosan-microplastics.html"].date == "2026-05-30"
    assert refs["chitosan-microplastics.html"].listing_url == BASE_URL + "archives2026.html"
    # no space between </a> and the date
    assert refs["astragin-ulcerative-colitis.html"].date == "2026-05-02"
    # stray </a> after the date
    assert refs["strength-training-cognitive-decline.html"].date == "2026-05-06"
    assert all(ref.date for ref in refs.values())


def test_listing_2009_unquoted_hrefs():
    refs = refs_by_slug("archives2009.html")
    assert len(refs) > 100
    assert refs["q10fat.html"].date == "2009-05-31"
    assert all(ref.date and ref.date.startswith("2009-") for ref in refs.values())


def test_listing_excludes_navigation():
    for name in ("archives2026.html", "archives2009.html", "index.html"):
        for link in refs_by_slug(name):
            assert not link.startswith("archives")
            assert link not in {
                "index.html",
                "us.html",
                "contact.html",
                "ergo-log-newsletter.html",
            }
            assert "/" not in link and link.endswith(".html")


def test_listing_index_page():
    refs = refs_by_slug("index.html")
    # featured articles at the top (headline + "(DD.MM.YYYY)") ...
    assert refs["inulin-pain-worn-joints.html"].date == "2026-09-29"
    assert refs["magnesium-l-threonate-sleep.html"].date == "2026-07-19"
    # ... and the plain list below them (June/July 2026 are not on any archive page yet)
    assert refs["luteolin-sleep.html"].date == "2026-07-07"
    assert len(refs) >= 50
    assert all(ref.date for ref in refs.values())


# --- articles ---------------------------------------------------------------------------------------------------

RELATED_TITLES = {
    "q10fat.html": [
        "Q10 and selenium supplements halve chance",  # "More:" block
        "Q10 supplementation makes explosive movements",  # related-articles table
        "Does Q10 offer strength athletes cardiac protection",
    ],
    "chitosan-microplastics.html": [
        "How to remove microplastics from drinking water",
        "This is how you remove heavy metals",
        "How to remove PFAS from your body",
    ],
}


@pytest.mark.parametrize(
    ("name", "title", "date", "categories", "body_snippets"),
    [
        (
            "q10fat.html",
            "Q10 combined with light exercise burns more fat",
            "2009-05-31",
            ["Weight Loss", "Q10"],
            [
                "After taking a capsule containing 30 mg",
                # text after an in-paragraph <table> (lxml closes the <p> there)
                "The co-enzyme Q10 is a molecular thumb tack",
                "Study\nUntil now researchers",
                "control group – which had been given a placebo –",
            ],
        ),
        (
            "chitosan-microplastics.html",
            "Chitosan blocks the absorption of microplastics from food",
            "2026-05-30",
            ["Miscellaneous"],
            [
                "The plastics that we dump into the environment",
                "Bottled water ranks first.",
                "Mechanism\nThe researchers suspect",
            ],
        ),
    ],
)
def test_parse_article(name, title, date, categories, body_snippets):
    article = parse_article(load(name), BASE_URL + name)
    assert article is not None
    assert article.link == BASE_URL + name
    assert article.title == title
    assert article.date == date
    assert article.categories == categories

    text = article.text
    for snippet in body_snippets:
        assert snippet in text
    for marker in ("More:", "Source:", "Archives:", "\x96", "�"):
        assert marker not in text
    for related in RELATED_TITLES[name]:
        assert related not in text
    assert not text.startswith(title)  # the title is not repeated as the first line
    assert text == text.strip()
    assert "\n\n\n" not in text
    assert not re.search(r"[ \t]{2,}", text)


def test_parse_article_title_with_inline_tags():
    html = '<table><tr><td><p id="kopindex">Is DMHA present in <i>Kigelia africana</i>?</p>Body.</td></tr></table>'
    assert parse_article(html, BASE_URL + "x.html").title == "Is DMHA present in Kigelia africana?"


def test_parse_article_returns_none_for_non_articles():
    assert parse_article("<html><body><h1>404 Not Found</h1></body></html>", BASE_URL + "x.html") is None
    varnish = "<html><head><title>503 Backend fetch failed</title></head><body><h1>Error 503</h1></body></html>"
    assert parse_article(varnish, BASE_URL + "x.html") is None
    assert parse_article('<p id="kopindex">Title only</p><p id=bodytekst3>Source:<br></p>', BASE_URL) is None
