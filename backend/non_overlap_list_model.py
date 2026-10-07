import heapq
from collections import defaultdict

from .preprocessing import tokenize


class NonOverlapListModel:
    """Search disjoint metadata-field regions with term posting lists.

    Each field is an independent region. Unlike OverlapListModel, this model
    does not create a parent record region containing the child fields.
    """

    REGION_FIELDS = (
        ("label", "label"),
        ("description", "description"),
        ("severity", "severity"),
        ("asd_status", "asd_status"),
        ("tags", "tags"),
        ("modalities", "modalities"),
    )

    def __init__(self, documents):
        self.documents = documents
        self.postings = defaultdict(lambda: defaultdict(set))
        self.regions_by_id = {}

        for document in documents:
            doc_id = document["document_id"]
            metadata = document.get("metadata", {})
            regions = {}
            for region_name, field_name in self.REGION_FIELDS:
                value = metadata.get(field_name)
                if value in (None, ""):
                    continue
                terms = set(tokenize(str(value)))
                if not terms:
                    continue
                regions[region_name] = terms
                for term in terms:
                    self.postings[term][region_name].add(doc_id)
            self.regions_by_id[doc_id] = regions

    def search(self, query, top_k=10, modality="all"):
        query_terms = set(tokenize(query))
        if not query_terms:
            raise ValueError("Enter a search query containing searchable terms.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        candidates = set()
        for term in query_terms:
            for document_ids in self.postings.get(term, {}).values():
                candidates.update(document_ids)

        ranked = []
        for document in self.documents:
            doc_id = document["document_id"]
            if doc_id not in candidates:
                continue
            if modality and modality.lower() not in {"all", "sample"}:
                if not document["files"].get(modality.lower(), {}).get("exists"):
                    continue

            regions = self.regions_by_id[doc_id]
            matched_regions = sorted(
                name for name, terms in regions.items() if terms & query_terms
            )
            matched_terms = sorted(
                set().union(*(regions[name] for name in matched_regions)) & query_terms
            ) if matched_regions else []
            if not matched_terms:
                continue

            ranked.append({
                **document,
                "score": len(matched_regions),
                "matched_terms": matched_terms,
                "matched_regions": matched_regions,
                "region_match_count": len(matched_regions),
            })

        # A field-coverage ordering helps compare results; it is not a
        # probabilistic or canonical relevance score for the list model.
        return heapq.nlargest(
            top_k,
            ranked,
            key=lambda result: (
                result["region_match_count"],
                len(result["matched_terms"]),
                result["document_id"],
            ),
        )
