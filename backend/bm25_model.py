import heapq
import math
from collections import Counter

from .preprocessing import tokenize


class BM25Model:
    """Okapi BM25 ranking with term-frequency saturation and length normalization."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(document["tokens"]) for document in documents]
        self.doc_lengths = [len(document["tokens"]) for document in documents]
        self.document_frequency = Counter(
            term for counts in self.term_counts for term in counts
        )
        self.document_count = len(documents)
        self.average_length = (
            sum(self.doc_lengths) / self.document_count if self.document_count else 0
        )

    def _idf(self, term):
        frequency = self.document_frequency[term]
        return math.log(
            1 + (self.document_count - frequency + 0.5) / (frequency + 0.5)
        )

    def search(self, query, top_k=10, modality="all"):
        query_terms = set(tokenize(query))
        if not query_terms:
            raise ValueError("Enter a search query containing searchable terms.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        ranked = []
        for document, counts, length in zip(
            self.documents, self.term_counts, self.doc_lengths
        ):
            if modality and modality.lower() not in {"all", "sample"}:
                if not document["files"].get(modality.lower(), {}).get("exists"):
                    continue

            matched_terms = query_terms & counts.keys()
            if not matched_terms:
                continue

            length_ratio = (
                length / self.average_length if self.average_length else 0
            )
            normalization = self.k1 * (1 - self.b + self.b * length_ratio)
            score = 0.0
            for term in matched_terms:
                term_frequency = counts[term]
                saturated_tf = (
                    term_frequency * (self.k1 + 1) / (term_frequency + normalization)
                )
                score += self._idf(term) * saturated_tf

            ranked.append(
                {
                    **document,
                    "score": score,
                    "matched_terms": sorted(matched_terms),
                }
            )

        return heapq.nsmallest(
            top_k,
            ranked,
            key=lambda result: (-result["score"], result["document_id"]),
        )
