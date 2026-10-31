"""Step 2: load kb/*.md, chunk by [SECTION-ID], and retrieve relevant chunks.

Retrieval uses TF-IDF cosine similarity (scikit-learn) so it runs fully offline
with no embedding API calls or model downloads. Swap `_vectorize` for a hosted
embedding model later if you want semantic (not just lexical) retrieval.
"""
import os
import re
from dataclasses import dataclass, field
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from src.config import KB_DIR
except ImportError:  # running as a script from inside src/
    from config import KB_DIR

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
SECTION_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.M)

DOC_TYPE_RANK = {"policy": 0, "agent_guide": 1, "help": 2, "faq": 3, "marketing": 4, "meta": 5}

# Customers don't use the documents' words (see README's "Things that make this
# realistic"). Expand common customer phrasing to the vocabulary the kb actually
# uses, so lexical (TF-IDF) retrieval still finds the right section.
QUERY_SYNONYMS = {
    "shoes": "footwear clothing", "shoe": "footwear",
    "clothes": "clothing fashion", "cloth": "clothing",
    "send it back": "return reverse pickup", "send back": "return",
    "nobody came to collect it": "reverse pickup missed",
    "did not come": "missed pickup", "didn't come": "missed pickup",
    "hasn't arrived": "not delivered late", "not arrived": "not delivered",
    "cancel": "cancellation", "money back": "refund",
    "broken": "damaged defective", "not working": "damaged defective troubleshooting",
    "wrong item": "wrong item missing", "swollen": "swelling safety",
    "exchange": "exchange return", "warranty claim": "warranty assured",
    "cod": "cash on delivery", "track": "tracking status",
    "never placed": "did not make unauthorized account activity",
    "didn't place": "did not make unauthorized account activity",
    "not mine": "did not make unauthorized account activity",
    "refund come": "how long refund take timeline business days initiated",
    "when will my refund": "how long refund take timeline business days initiated",
    "when will i get my refund": "how long refund take timeline business days initiated",
    "where is my refund": "how long refund take timeline business days initiated",
    "collection": "reverse pickup", "torn": "damaged defective", "ripped": "damaged defective",
    "already shipped": "after the order ships shipped cancellation",
    "has shipped": "after the order ships shipped cancellation",
    "already out for delivery": "after the order ships shipped cancellation",
}

# Extra terms to nudge retrieval toward the RIGHT kind of policy section once a
# product's return_category is known (RET-2 names the category codes, but the
# *condition* rules like "unopened" live in separate sections with different
# wording -- this bridges that gap).
CATEGORY_EXTRA_TERMS = {
    "beauty_sealed": "unopened seal intact condition",
    "electronics": "screen lock removed physically damaged condition",
    "fashion": "unused tags packaging condition",
    "innerwear": "not returnable change of mind",
    "digital": "not returnable",
    "grocery": "not returnable",
}


HINDI_SYNONYMS = {
    # customers write in Hindi/Hinglish while most docs are English -- the kb has
    # no Devanagari policy text to match against, so bridge common terms to the
    # English vocabulary the kb actually uses.
    "वापस": "return", "लौटा": "return", "लौटाने": "return", "लौट": "return",
    "दिन": "days window", "कितने": "how many",
    "रिफंड": "refund", "रिफ़ंड": "refund", "पैसे": "refund money",
    "कैंसिल": "cancel cancellation", "रद्द": "cancel cancellation",
    "ऑर्डर": "order", "डिलीवरी": "delivery", "शिपिंग": "shipping",
    "खराब": "damaged defective", "गलत": "wrong", "टूटा": "damaged broken",
    "खोल": "opened unsealed", "वारंटी": "warranty",
}


def _load_products():
    """title, sku, return_category, price_inr -- used to connect a product
    mentioned by name/SKU in a question to its policy-relevant catalog facts
    (category code, high-value threshold), since the kb never names products."""
    import csv
    path = os.path.join(os.path.dirname(KB_DIR), "data", "products.csv")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


_PRODUCTS = None
load_products = _load_products  # public alias, used by agent.py too


def _catalog_terms(query: str) -> str:
    global _PRODUCTS
    if _PRODUCTS is None:
        _PRODUCTS = _load_products()
    extra = []
    q_lower = query.lower()
    for p in _PRODUCTS:
        title = p["title"]
        base = title.split(",")[0].strip()  # drop ", Blue" / ", 256 GB" variant suffixes
        key = " ".join(base.split()[:4]).lower()
        if (title.lower() in q_lower or base.lower() in q_lower
                or (len(key) > 8 and key in q_lower) or p["sku"] in query):
            cat = p["return_category"]
            extra.append(cat)
            extra.append(CATEGORY_EXTRA_TERMS.get(cat, ""))
            if p.get("assured") == "True":
                extra.append("assured warranty")
            try:
                if int(p["price_inr"]) > 10000:
                    extra.append("high-value more than 10000 priority resolution")
            except (ValueError, TypeError):
                pass
    return " ".join(extra)


def _expand_query(query: str) -> str:
    q = query.lower()
    extra = [exp for phrase, exp in QUERY_SYNONYMS.items() if phrase in q]
    for phrase, exp in HINDI_SYNONYMS.items():
        if phrase in query:  # Devanagari isn't lowercased by .lower()
            extra.append(exp)
    extra.append(_catalog_terms(query))
    return query + " " + " ".join(extra)


@dataclass
class Chunk:
    section_id: str
    heading: str
    text: str
    doc_id: str
    title: str
    doc_type: str
    status: str
    effective_from: str
    source_file: str


def _parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def load_chunks() -> List[Chunk]:
    chunks = []
    for fname in sorted(os.listdir(KB_DIR)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(KB_DIR, fname)
        raw = open(path, encoding="utf-8").read()
        meta = _parse_frontmatter(raw)
        body = FRONTMATTER_RE.sub("", raw, count=1)

        matches = list(SECTION_RE.finditer(body))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            section_text = body[start:end].strip()
            chunks.append(Chunk(
                section_id=m.group(1).strip(),
                heading=m.group(2).strip(),
                text=f"{m.group(2).strip()}\n{section_text}",
                doc_id=meta.get("doc_id", fname),
                title=meta.get("title", fname),
                doc_type=meta.get("doc_type", "unknown"),
                status=meta.get("status", "current"),
                effective_from=meta.get("effective_from", ""),
                source_file=fname,
            ))
    return chunks


class KBIndex:
    def __init__(self):
        self.chunks = load_chunks()
        corpus = [f"{c.title} {c.heading} {c.text}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 6) -> List[Chunk]:
        qvec = self.vectorizer.transform([_expand_query(query)])
        sims = cosine_similarity(qvec, self.matrix)[0]
        # A short superseded chunk can out-score a longer current one purely on
        # TF-IDF term weight. Apply a real penalty (not just a tie-break) so
        # current policy/agent_guide docs are favoured over superseded/marketing
        # ones at comparable relevance, per kb/00_about_this_knowledge_base.md
        # [KB-00.1]. Superseded chunks still surface when they're clearly the
        # best (or only) match, e.g. for an item delivered under the old policy.
        def boosted(i):
            c = self.chunks[i]
            score = sims[i]
            if c.status != "current":
                score *= 0.7
            score *= 1.0 - 0.02 * DOC_TYPE_RANK.get(c.doc_type, 9)
            return score

        ranked = sorted(range(len(self.chunks)), key=lambda i: -boosted(i))
        top = [i for i in ranked if sims[i] > 0][:k]
        return [self.chunks[i] for i in top]


if __name__ == "__main__":
    idx = KBIndex()
    print(f"Loaded {len(idx.chunks)} chunks from {len(set(c.source_file for c in idx.chunks))} docs")
    for c in idx.retrieve("how long for a refund after cancellation", k=3):
        print(f"  [{c.section_id}] {c.heading}  ({c.doc_type}, {c.status})")
