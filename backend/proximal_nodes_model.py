import heapq
import re
import unicodedata
from collections import defaultdict

from .preprocessing import STOPWORDS, tokenize


class ProximalNodesModel:
    """Hierarchical field/sentence retrieval with term-position proximity."""

    REGION_FIELDS = (
        ("label", "label"),
        ("description", "description"),
        ("severity", "severity"),
        ("asd_status", "asd_status"),
        ("tags", "tags"),
        ("modalities", "modalities"),
    )
    TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.I)
    SENTENCE_RE = re.compile(r"[^.!?;]+(?:[.!?;]+|$)")

    def __init__(self, documents):
        self.documents = documents
        self.nodes_by_id = {}
        self.parents_by_id = {}
        self.postings = defaultdict(lambda: defaultdict(set))

        for document in documents:
            doc_id = document["document_id"]
            metadata = document.get("metadata", {})
            nodes = {}
            parents = {}
            record_positions = defaultdict(list)
            record_offset = 0

            for field_name, metadata_key in self.REGION_FIELDS:
                value = metadata.get(metadata_key)
                if value in (None, ""):
                    continue
                text = unicodedata.normalize("NFKC", str(value)).lower()
                positioned = self._positioned_tokens(text)
                if not positioned:
                    continue

                field_id = field_name
                field_positions = defaultdict(list)
                for term, position in positioned:
                    field_positions[term].append(position)
                    record_positions[term].append(record_offset + position)
                nodes[field_id] = dict(field_positions)
                parents[field_id] = "record"

                if field_name == "description":
                    # Sentence nodes are children of the description field.
                    for sentence_index, match in enumerate(self.SENTENCE_RE.finditer(text)):
                        start, end = match.span()
                        sentence_terms = self._positioned_tokens(match.group())
                        if not sentence_terms:
                            continue
                        sentence_id = f"description.sentence.{sentence_index}"
                        sentence_positions = defaultdict(list)
                        for term, local_position in sentence_terms:
                            sentence_positions[term].append(local_position)
                        nodes[sentence_id] = dict(sentence_positions)
                        parents[sentence_id] = field_id

                record_offset += max((position for _, position in positioned), default=0) + 2

            if record_positions:
                nodes["record"] = dict(record_positions)
            self.nodes_by_id[doc_id] = nodes
            self.parents_by_id[doc_id] = parents
            for node_id, terms_to_positions in nodes.items():
                for term in terms_to_positions:
                    self.postings[term][doc_id].add(node_id)

    @classmethod
    def _positioned_tokens(cls, text):
        normalized = unicodedata.normalize("NFKC", str(text)).lower()
        return [
            (match.group(), index)
            for index, match in enumerate(cls.TOKEN_RE.finditer(normalized))
            if match.group() not in STOPWORDS
        ]

    @staticmethod
    def _best_window(query_terms, positions_by_term):
        """Return the smallest positional span containing every query term."""
        if any(not positions_by_term.get(term) for term in query_terms):
            return None
        entries = sorted(
            (position, term)
            for term in query_terms
            for position in positions_by_term[term]
        )
        counts = defaultdict(int)
        covered = 0
        left = 0
        best = None
        for right, (position, term) in enumerate(entries):
            if counts[term] == 0:
                covered += 1
            counts[term] += 1
            while covered == len(query_terms):
                width = position - entries[left][0]
                best = width if best is None else min(best, width)
                left_term = entries[left][1]
                counts[left_term] -= 1
                if counts[left_term] == 0:
                    covered -= 1
                left += 1
        return best

    def search(self, query, top_k=10, modality="all"):
        query_terms = set(tokenize(query))
        if not query_terms:
            raise ValueError("Enter a search query containing searchable terms.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        candidates = set()
        for term in query_terms:
            candidates.update(self.postings.get(term, {}).keys())

        ranked = []
        for document in self.documents:
            doc_id = document["document_id"]
            if doc_id not in candidates:
                continue
            if modality and modality.lower() not in {"all", "sample"}:
                if not document["files"].get(modality.lower(), {}).get("exists"):
                    continue

            nodes = self.nodes_by_id[doc_id]
            node_matches = []
            all_matched_terms = set()
            for node_id, positions in nodes.items():
                # The root spans unrelated fields; field-local positions are
                # the meaningful units for proximity comparison.
                if node_id == "record":
                    continue
                matched = query_terms & positions.keys()
                if not matched:
                    continue
                all_matched_terms.update(matched)
                window = self._best_window(matched, positions)
                node_matches.append({
                    "node": node_id,
                    "terms": matched,
                    "coverage": len(matched),
                    "window": window,
                })

            if not node_matches:
                continue
            # Prefer a sentence containing more query terms, then a tighter
            # window. Field nodes remain as fallback for non-description text.
            best_node = min(
                node_matches,
                key=lambda item: (
                    -item["coverage"],
                    item["window"] if item["window"] is not None else float("inf"),
                    0 if ".sentence." in item["node"] else 1,
                    item["node"],
                ),
            )
            region_hits = [
                {
                    "node": item["node"],
                    "parent": self.parents_by_id[doc_id].get(item["node"]),
                    "matched_terms": sorted(item["terms"]),
                    "window": item["window"],
                }
                for item in node_matches
                if item["coverage"] == best_node["coverage"]
                and item["window"] == best_node["window"]
            ]
            ranked.append({
                **document,
                "score": best_node["coverage"] / (1 + (best_node["window"] or 0)),
                "matched_terms": sorted(all_matched_terms),
                "matched_regions": region_hits,
                "proximity_window": best_node["window"],
                "proximity_node": best_node["node"],
                "proximity_parent": self.parents_by_id[doc_id].get(best_node["node"]),
                "term_coverage": best_node["coverage"],
            })

        return heapq.nlargest(
            top_k,
            ranked,
            key=lambda result: (
                result["term_coverage"],
                -(result["proximity_window"] if result["proximity_window"] is not None else 10**9),
                len(result["matched_terms"]),
                result["document_id"],
            ),
        )
