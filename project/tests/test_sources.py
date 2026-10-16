from functions import format_docs, format_sources, rewrite_citations
from sample_docs import BANANA_LINK, DOCS, EXAMPLE_OUTPUT, GINGER_LINK, GREEN_TEA_LINK


def test_dedupes_by_link_preserving_order():
    lines = format_sources(DOCS).splitlines()

    assert len(lines) == 5  # 6 chunks, two from the same article
    assert GREEN_TEA_LINK in lines[0]
    assert GINGER_LINK in lines[1]
    assert sum(GREEN_TEA_LINK in line for line in lines) == 1


def test_numbered_markdown_links_with_dates():
    lines = format_sources(EXAMPLE_OUTPUT["context"]).splitlines()

    assert lines == [
        f"1. [Green tea extract boosts fat burning during exercise]({GREEN_TEA_LINK}) — 14-03-2010",
        f"2. [Ginger reduces muscle pain after strength training]({GINGER_LINK}) — 02-09-2012",
    ]


def test_missing_date_is_omitted():
    banana = next(doc for doc in DOCS if doc.metadata["article_link"] == BANANA_LINK)

    assert format_sources([banana]) == f"1. [Bananas as good as sports drinks for cyclists]({BANANA_LINK})"


def test_empty():
    assert format_sources([]) == ""


def test_format_docs_for_prompt():
    text = format_docs(DOCS[:2])  # two chunks, same article → one link
    title = "Green tea extract boosts fat burning during exercise"

    assert text.startswith("Links (copy exactly):\n")
    assert text.count(f"- [{title}]({GREEN_TEA_LINK})") == 1
    assert f"### {title} (2010-03-14)\n" in text
    assert "EGCG" in text


def test_rewrite_citations_fixes_invented_url():
    docs = DOCS[:1]
    title = docs[0].metadata["article_title"]
    invented = "https://www.ergo-log.com/green-tea-extract-boosts-fat-burning-slugified-wrong.html"
    answer = f"- Point. [{title}]({invented})"

    fixed = rewrite_citations(answer, docs)

    assert GREEN_TEA_LINK in fixed
    assert invented not in fixed


def test_rewrite_citations_matches_title_prefix():
    docs = DOCS[:1]
    title = docs[0].metadata["article_title"]
    invented = "https://www.ergo-log.com/made-up-slug.html"
    answer = f"- Point. [{title[:20]}]({invented})"

    fixed = rewrite_citations(answer, docs)

    assert f"[{title}]({GREEN_TEA_LINK})" in fixed
