"""Example documents and chain output for app tests (no index or API key needed)."""

from langchain_core.documents import Document

GREEN_TEA_LINK = "https://www.ergo-log.com/green-tea-extract-boosts-fat-burning-during-exercise.html"
GINGER_LINK = "https://www.ergo-log.com/ginger-reduces-muscle-pain-after-strength-training.html"
BANANA_LINK = "https://www.ergo-log.com/bananas-as-good-as-sports-drinks-for-cyclists.html"
CREATINE_LINK = "https://www.ergo-log.com/creatine-plus-strength-training-more-muscle-mass.html"
BEETROOT_LINK = "https://www.ergo-log.com/beetroot-juice-improves-endurance.html"


def _doc(title: str, link: str, date: str, content: str, chunk: int = 0, **extra) -> Document:
    metadata = {
        "article_title": title,
        "article_link": link,
        "article_date": date,
        "archive_link": "https://www.ergo-log.com/archive.html",
        "categories": extra.pop("categories", "Supplements"),
        "chunk": chunk,
    }
    return Document(page_content=content, metadata=metadata)


DOCS: list[Document] = [
    _doc(
        "Green tea extract boosts fat burning during exercise",
        GREEN_TEA_LINK,
        "2010-03-14",
        "Subjects who took green tea extract burned 17 percent more fat during moderate cycling. "
        "The researchers attribute the effect to the catechin EGCG.",
    ),
    _doc(
        "Green tea extract boosts fat burning during exercise",
        GREEN_TEA_LINK,
        "2010-03-14",
        "EGCG inhibits the enzyme that breaks down noradrenaline, which keeps fat cells releasing fatty acids longer.",
        chunk=1,
    ),
    _doc(
        "Ginger reduces muscle pain after strength training",
        GINGER_LINK,
        "2012-09-02",
        "Two grams of ginger a day for eleven days reduced muscle pain after eccentric exercise by about 25 percent.",
        categories="Herbs;Recovery",
    ),
    _doc(
        "Bananas as good as sports drinks for cyclists",
        BANANA_LINK,
        "",
        "Cyclists who ate bananas during a 75 km time trial performed as well as those drinking a carbohydrate drink.",
        categories="Nutrition",
    ),
    _doc(
        "Creatine plus strength training means more muscle mass",
        CREATINE_LINK,
        "2015-06-21",
        "Creatine supplementation combined with strength training increased lean body mass more than training alone.",
    ),
    _doc(
        "Beetroot juice improves endurance",
        BEETROOT_LINK,
        "2009-11-30",
        "Nitrate from beetroot juice lowered the oxygen cost of exercise and extended time to exhaustion by 16%.",
        categories="Nutrition;Endurance",
    ),
]

EXAMPLE_OUTPUT: dict = {
    "question": "What are health benefits of green tea and why?",
    "context": DOCS[:3],
    "answer": (
        "- Green tea extract increases fat burning during exercise (about 17% more fat in one study).\n"
        "- The catechin EGCG keeps noradrenaline active for longer, so fat cells keep releasing fatty acids."
    ),
}
