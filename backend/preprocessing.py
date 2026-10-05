import re
import unicodedata

# Keep medically meaningful terms (including "no" and "not") searchable.
STOPWORDS = frozenset("a an and are as at be been being by for from has have in into is it its of on or that the their this to was were with".split())


def tokenize(value):
    if value is None:
        return []
    text = unicodedata.normalize("NFKC", str(value)).lower()
    # Retain alphanumeric medical terms and identifiers; normalize separators.
    return [t for t in re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text) if t not in STOPWORDS]


def normalize_text(value):
    return " ".join(tokenize(value))
