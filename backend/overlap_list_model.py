import heapq
from collections import defaultdict

from .preprocessing import tokenize


class OverlapListModel:
    """Search nested document regions using overlapping term posting lists.

    Each sample is a parent region containing named child regions. A term can
    therefore occur in both its field region and the enclosing sample region.
    Child regions remain separate so results can explain where terms matched.
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
            child_regions = {}
            for region_name, field_name in self.REGION_FIELDS:
                value = metadata.get(field_name)
                if value in (None, ""):
                    continue
                tokens = set(tokenize(str(value)))
                if tokens:
                    child_regions[region_name] = tokens
                    for term in tokens:
                        self.postings[term][region_name].add(doc_id)

            # The parent region overlaps every child region by containment.
            parent_tokens = set().union(*child_regions.values()) if child_regions else set()
            self.regions_by_id[doc_id] = {
                "parent": parent_tokens,
                "children": child_regions,
            }
            for term in parent_tokens:
                self.postings[term]["record"].add(doc_id)

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
            matched_fields = sorted(
                name
                for name, terms in regions["children"].items()
                if terms & query_terms
            )
            matched_terms = sorted(query_terms & regions["parent"])
            if not matched_terms:
                continue

            # This is an explainable region-coverage ordering, not a
            # probabilistic or BM25 relevance score.
            ranked.append({
                **document,
                "score": len(matched_fields),
                "matched_terms": matched_terms,
                "matched_regions": (["record"] if matched_terms else []) + matched_fields,
                "region_match_count": len(matched_fields),
            })

        return heapq.nlargest(
            top_k,
            ranked,
            key=lambda result: (
                result["region_match_count"],
                len(result["matched_terms"]),
                result["document_id"],
            ),
        )
