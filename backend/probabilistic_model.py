import heapq
import math
from collections import Counter

from .preprocessing import tokenize


class ProbabilisticModel:
    """Smoothed query-likelihood retrieval with posterior-probability ranking."""

    def __init__(self, documents, smoothing=1.0):
        self.documents = documents
        self.smoothing = smoothing
        self.term_counts = [Counter(document["tokens"]) for document in documents]
        self.doc_lengths = [len(document["tokens"]) for document in documents]
        self.vocabulary = {
            term for counts in self.term_counts for term in counts
        }

    def _query_log_likelihood(self, query_counts, counts, document_length, vocabulary_size):
        """Return log P(query | document), with Laplace smoothing."""
        denominator = document_length + self.smoothing * vocabulary_size
        return sum(
            frequency
            * math.log(
                (counts.get(term, 0) + self.smoothing) / denominator
            )
            for term, frequency in query_counts.items()
        )

    def search(self, query, top_k=10, modality="all"):
        query_terms = tokenize(query)
        if not query_terms:
            raise ValueError("Enter a search query containing searchable terms.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_counts = Counter(query_terms)
        unique_terms = set(query_counts)
        vocabulary_size = len(self.vocabulary | unique_terms)
        candidates = []

        for document, counts, length in zip(
            self.documents, self.term_counts, self.doc_lengths
        ):
            if modality and modality.lower() not in {"all", "sample"}:
                if not document["files"].get(modality.lower(), {}).get("exists"):
                    continue

            matched_terms = unique_terms & counts.keys()
            if not matched_terms:
                continue

            log_likelihood = self._query_log_likelihood(
                query_counts, counts, length, vocabulary_size
            )
            candidates.append((log_likelihood, document, sorted(matched_terms)))

        if not candidates:
            return []

        # Bayes with a uniform document prior. Subtracting the maximum log
        # likelihood keeps the exponentials stable without changing the order.
        max_log_likelihood = max(item[0] for item in candidates)
        likelihoods = [math.exp(item[0] - max_log_likelihood) for item in candidates]
        evidence = sum(likelihoods)
        posterior_results = []
        for (_, document, matched_terms), likelihood in zip(candidates, likelihoods):
            probability = likelihood / evidence
            posterior_results.append(
                {
                    **document,
                    "score": probability,
                    "probability": probability,
                    "matched_terms": matched_terms,
                }
            )

        # Keep the requested highest probabilities in a heap-backed top-K queue.
        return heapq.nsmallest(
            top_k,
            posterior_results,
            key=lambda result: (-result["probability"], result["document_id"]),
        )
