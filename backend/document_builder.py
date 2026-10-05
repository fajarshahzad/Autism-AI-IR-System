import re
from pathlib import Path
from .preprocessing import tokenize

MODALITIES = ("image", "voice", "motion", "physio")


def build_documents(root, records):
    root = Path(root)
    docs = []
    for row in records:
        doc_id = str(row.get("sample_id") or row.get("id") or "").strip()
        if not doc_id:
            continue
        files = {}
        for modality in MODALITIES:
            rel = str(row.get(modality) or "").strip()
            path = (root / rel).resolve() if rel else None
            # Image reference discrepancy in this dataset: resolve Child (N).jpg
            # from the numeric sample suffix only, while retaining the source path.
            if modality == "image" and path and not path.exists():
                match = re.search(r"(\d+)$", doc_id)
                candidate = root / "images" / f"Child ({int(match.group(1))}).jpg" if match else None
                if candidate and candidate.is_file():
                    path = candidate.resolve()
            indexed_candidate = (root / rel).resolve() if rel else None
            files[modality] = {"indexed_path": rel or None,
                               "indexed_path_exists": bool(indexed_candidate and indexed_candidate.is_file()),
                               "resolution": "numeric_sample_filename_inference" if modality == "image" and rel and not (root / rel).is_file() and path and path.is_file() else "indexed_path",
                               "path": str(path) if path and path.is_file() else None,
                               "exists": bool(path and path.is_file()),
                               "filename": path.name if path and path.is_file() else (Path(rel).name if rel else None),
                               "file_type": path.suffix.lower().lstrip(".") if path and path.is_file() else (Path(rel).suffix.lower().lstrip(".") if rel else None)}
        text_fields = [row.get(k) for k in ("sample_id", "label", "description", "severity", "asd_status", "tags", "search_text", "modalities")]
        # Include modality labels and names, not numeric signal arrays.
        searchable = " ".join(str(v) for v in text_fields if v not in (None, "")) + " " + " ".join(MODALITIES)
        keywords = [x.strip() for x in str(row.get("tags") or "").split(";") if x.strip()]
        docs.append({"document_id": doc_id, "filename": doc_id, "modality": "sample", "modalities": list(MODALITIES),
                     "searchable_text": searchable, "tokens": tokenize(searchable),
                     "description": row.get("description") or "", "keywords": keywords,
                     "label": row.get("label"), "path": files, "metadata": row, "files": files})
    return docs
