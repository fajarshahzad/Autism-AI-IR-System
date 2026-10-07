import heapq
import math
from collections import Counter

from .preprocessing import tokenize


class BinaryIndependenceModel:
    """Binary term-presence retrieval using Robertson-Sparck Jones weights.

    With no feedback, the model uses a collection-frequency initial estimate.
    Optional query-specific relevant/non-relevant IDs update the term odds.
    """

    def __init__(self, documents):
        self.documents = documents
        self.by_id = {document["document_id"]: document for document in documents}
        self.terms_by_id = {
            document["document_id"]: set(document["tokens"]) for document in documents
        }
        self.term_sets = [set(document["tokens"]) for document in documents]
        self.document_frequency = Counter(
            term for terms in self.term_sets for term in terms
        )

    def _feedback_weights(self, terms, relevant_ids, non_relevant_ids):
        total_docs = len(self.documents)
        relevant_ids = set(relevant_ids or [])
        non_relevant_ids = set(non_relevant_ids or [])

        unknown_ids = (relevant_ids | non_relevant_ids) - self.by_id.keys()
        if unknown_ids:
            raise ValueError(f"Unknown feedback document ID: {sorted(unknown_ids)[0]}")
        overlap = relevant_ids & non_relevant_ids
        if overlap:
            raise ValueError("A document cannot be marked both relevant and non-relevant.")

        relevant_sets = [self.terms_by_id[doc_id] for doc_id in relevant_ids]
        non_relevant_sets = [self.terms_by_id[doc_id] for doc_id in non_relevant_ids]
        weights = {}

        for term in terms:
            collection_df = self.document_frequency[term]
            # Collection-based starting estimates are used until both classes
            # have feedback. A 0.5 correction prevents zero/one probabilities.
            base_u = (collection_df + 0.5) / (total_docs + 1)
            p = 0.5
            u = base_u

            if relevant_sets:
                r = sum(term in doc_terms for doc_terms in relevant_sets)
                p = (r + 0.5) / (len(relevant_sets) + 1)
            if non_relevant_sets:
                n = sum(term in doc_terms for doc_terms in non_relevant_sets)
                u = (n + 0.5) / (len(non_relevant_sets) + 1)

            weights[term] = math.log((p * (1 - u)) / (u * (1 - p)))

        return weights

    def search(
        self,
        query,
        top_k=10,
        modality="all",
        relevant_ids=None,
        non_relevant_ids=None,
    ):
        query_terms = tokenize(query)
        if not query_terms:
            raise ValueError("Enter a search query containing searchable terms.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        terms = set(query_terms)
        feedback_ids = set(relevant_ids or []) | set(non_relevant_ids or [])
        if modality and modality.lower() not in {"all", "sample"}:
            unavailable = [
                doc_id
                for doc_id in feedback_ids
                if not self.by_id.get(doc_id, {}).get("files", {}).get(
                    modality.lower(), {}
                ).get("exists")
            ]
            if unavailable:
                raise ValueError(
                    f"Feedback document {unavailable[0]} does not have the selected modality."
                )

        weights = self._feedback_weights(terms, relevant_ids, non_relevant_ids)
        ranked = []

        for document, document_terms in zip(self.documents, self.term_sets):
            if modality and modality.lower() not in {"all", "sample"}:
                if not document["files"].get(modality.lower(), {}).get("exists"):
                    continue

            matched_terms = terms & document_terms
            if not matched_terms:
                continue

            # Terms are binary features: each query term contributes at most once.
            score = sum(weights[term] for term in matched_terms)
            ranked.append(
                {
                    **document,
                    "score": score,
                    "matched_terms": sorted(matched_terms),
                }
            )

        return heapq.nlargest(
            top_k,
            ranked,
            key=lambda result: (result["score"], result["document_id"]),
        )
