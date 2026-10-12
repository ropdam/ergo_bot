"""Pure HTML parsing for ergo-log.com (no network access).

The site is hand-written HTML with sloppy markup (unquoted attributes, stray closing tags, ``<table>`` elements
inside ``<p>``), served as ``text/html`` without a charset. Older pages are windows-1252 encoded, newer ones UTF-8.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

BASE_URL = "https://www.ergo-log.com/"

_SITE_HOSTS = {"www.ergo-log.com", "ergo-log.com"}
_NON_ARTICLE_PAGES = {
    "index.html",
    "us.html",
    "contact.html",
    "ergo-log-newsletter.html",
}
_DATE_RE = re.compile(r"(\d{2})[.-](\d{2})[.-](\d{4})")

# Tags whose text never belongs to the article body.
_SKIP_TEXT_IN = {"script", "style", "noscript", "iframe", "head", "title"}
# Tags that start a new paragraph / line when walking the article body.
_PARAGRAPH_TAGS = {
    "p",
    "section",
    "div",
    "table",
    "tr",
    "ul",
    "ol",
    "li",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "footer",
}


@dataclass
class ArticleRef:
    """An article link as found on a listing page (monthly archive page or the home page)."""

    link: str
    date: str | None  # ISO YYYY-MM-DD
    listing_url: str


@dataclass
class Article:
    """A parsed article page."""

    link: str
    title: str
    date: str | None  # ISO YYYY-MM-DD
    text: str
    categories: list[str] = field(default_factory=list)


def decode(content: bytes) -> str:
    """Decode raw page bytes: UTF-8 when valid, otherwise windows-1252 (the server never sends a charset)."""
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("windows-1252", errors="replace")


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _collapse(text: str) -> str:
    return " ".join(text.split())


def parse_date(text: str | None) -> str | None:
    """Return the first ``DD.MM.YYYY`` / ``DD-MM-YYYY`` date in ``text`` as ISO ``YYYY-MM-DD`` (None if invalid)."""
    if not text:
        return None
    for match in _DATE_RE.finditer(text):
        day, month, year = (int(g) for g in match.groups())
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue
    return None


def normalize_url(href: str, page_url: str) -> str | None:
    """Resolve ``href`` against ``page_url``, drop the fragment and canonicalise the ergo-log host."""
    href = href.strip()
    if not href or href.startswith(("mailto:", "javascript:", "#")):
        return None
    parts = urlsplit(urljoin(page_url, href))
    if parts.scheme not in {"http", "https"}:
        return None
    if parts.netloc.lower() in _SITE_HOSTS:
        base = urlsplit(BASE_URL)
        return urlunsplit((base.scheme, base.netloc, parts.path, parts.query, ""))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def is_article_url(url: str) -> bool:
    """True for ergo-log article pages (``https://www.ergo-log.com/<slug>.html``), False for navigation/archives."""
    parts = urlsplit(url)
    if parts.netloc != urlsplit(BASE_URL).netloc or parts.query:
        return False
    path = parts.path.lstrip("/")
    if "/" in path or not path.lower().endswith(".html"):
        return False
    name = path.lower()
    return not (name.startswith("archives") or name in _NON_ARTICLE_PAGES)


def parse_archive_index(html: str, base_url: str = BASE_URL) -> list[str]:
    """Monthly archive page URLs linked from ``archives.html`` (fragments stripped, de-duplicated, in page order)."""
    soup = _soup(html)
    urls: list[str] = []
    for a in soup.select("p#bodytekstindex a[href]"):
        url = normalize_url(a["href"], base_url)
        if url and url not in urls:
            urls.append(url)
    return urls


def _text_after(anchor: Tag) -> str:
    """Text following ``anchor`` in document order, up to the next link."""
    chunks: list[str] = []
    descendants = set(map(id, anchor.descendants))
    for el in anchor.next_elements:
        if id(el) in descendants:
            continue
        if isinstance(el, Tag) and el.name == "a":
            break
        if isinstance(el, NavigableString) and not isinstance(el, Comment):
            chunks.append(str(el))
            if sum(map(len, chunks)) > 200:
                break
    return "".join(chunks)


def parse_listing(html: str, page_url: str) -> list[ArticleRef]:
    """Article links on a listing page: a monthly archive page or ``index.html``.

    Archive pages list ``<a href=slug.html>TITLE</a> DD.MM.YYYY<br>`` lines inside ``p#bodytekst``; the home page also
    has featured articles (``p#kopindex`` headline followed by ``(DD.MM.YYYY)``) and a ``p#bodytekstindex`` list.
    Links inside a listing paragraph are always kept; other links only when a date follows them, which skips
    navigation, banners and the newsletter link. Duplicate links on one page keep their first occurrence.
    """
    soup = _soup(html)
    refs: dict[str, ArticleRef] = {}
    for a in soup.find_all("a", href=True):
        link = normalize_url(a["href"], page_url)
        if not link or not is_article_url(link) or link in refs:
            continue
        in_listing = a.find_parent("p", id=("bodytekst", "bodytekstindex", "kopindex")) is not None
        article_date = parse_date(_text_after(a))
        if article_date is None and not in_listing:
            continue
        refs[link] = ArticleRef(link=link, date=article_date, listing_url=page_url)
    return list(refs.values())


def _parse_article_date(soup: BeautifulSoup) -> str | None:
    for el in soup.find_all("p", id="wh"):
        found = parse_date(el.get_text())
        if found:
            return found
    return None


def _nearest(el: Tag | NavigableString, name: str) -> Tag | None:
    return el if isinstance(el, Tag) and el.name == name else el.find_parent(name)


def _body_text(title_el: Tag) -> str:
    """Article text: everything after the headline in the content cell, up to the first ``p#bodytekst3``.

    Walking the content cell in document order (instead of selecting ``p#bodytekst``) is needed because lxml closes
    a ``<p>`` when a ``<table>`` (ad iframe / image box) starts inside it, which leaves the rest of the paragraph as
    loose text in the cell. Text inside nested tables (ads, the related-articles grid) is skipped.
    """
    content_cell = title_el.find_parent("td")
    if content_cell is None:
        return ""
    title_nodes = {id(title_el), *map(id, title_el.descendants)}

    parts: list[str] = []
    started = False
    for el in content_cell.descendants:
        if el is title_el:
            started = True
            continue
        if not started or id(el) in title_nodes:
            continue
        if isinstance(el, Tag):
            if el.name == "p" and el.get("id") == "bodytekst3":
                break
            if el.name == "br":
                parts.append("\n")
            elif el.name in _PARAGRAPH_TAGS:
                parts.append("\n\n")
            continue
        if isinstance(el, Comment) or not isinstance(el, NavigableString):
            continue
        if el.parent is not None and el.parent.name in _SKIP_TEXT_IN:
            continue
        if _nearest(el, "td") is not content_cell:
            continue
        parts.append(re.sub(r"\s+", " ", str(el)))

    text = "".join(parts)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_article(html: str, url: str) -> Article | None:
    """Parse an article page. Returns None when there is no headline or the body is empty (e.g. an error page)."""
    soup = _soup(html)
    title_el = soup.find("p", id="kopindex")
    if title_el is None:
        return None
    title = _collapse(title_el.get_text())
    if not title:
        return None

    text = _body_text(title_el)
    if not text:
        return None

    categories: list[str] = []
    for block in soup.find_all("p", id="bodytekst3"):
        label = _collapse(block.get_text(" ", strip=True)).lower()
        if label.startswith("archives"):
            categories.extend(name for a in block.find_all("a") if (name := _collapse(a.get_text(" ", strip=True))))

    return Article(
        link=url,
        title=title,
        date=_parse_article_date(soup),
        text=text,
        categories=list(dict.fromkeys(categories)),
    )
