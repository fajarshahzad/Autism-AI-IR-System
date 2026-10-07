# Autism IR Search

A classical information retrieval demonstrator for the supplied multimodal autism sample dataset. It uses sample-level documents, Boolean retrieval over an inverted index, a TF-IDF vector space model with cosine similarity, a probabilistic query-likelihood model, a Binary Independence Model (BIM), BM25 ranking, overlapping/non-overlapping structured-region lists, and a Proximal Nodes model. It does not use embeddings, an LLM, or a vector database. The original dataset is read-only from the application's perspective.

## Dataset analysis

The supplied folder is `Dataset for autism copy/` and contains 403 filesystem entries: the index CSV, metadata JSON, 100 JPEGs, 100 motion JSON files, 100 physiological CSV files, 100 WAV files, plus a `.DS_Store`. The CSV has 100 rows and 14 columns: `sample_id`, `image`, `voice`, `motion`, `physio`, `label`, `description`, `severity_level`, `asd_status`, `severity`, `modalities`, `modality_count`, `tags`, and `search_text`. All CSV cells are populated; IDs and descriptions/search text are unique. There are no duplicate CSV rows. Metadata is a 100-element JSON array with the same 14 keys; severity_level and modality_count are JSON integers and CSV strings. Its IDs align row-for-row with the CSV.

Labels: mild_asd 23, moderate_asd 19, severe_asd 33, typical 25. Every index record references one image, voice, motion, and physio file. The 100 voice, motion, and physio references all exist. None of the 100 image references (`images/child_NNN.png`) exists; disk instead contains `images/Child (N).jpg` for N=1…100. The app transparently resolves these by the numeric sample suffix, retains the indexed path, and reports both paths. This is an inferred filename correspondence based on the complete one-to-one numbered sets; no image content was inspected or altered. The physiological files have columns `Time`, `HR`, `GSR`, `TEMP`; motion JSON has `head`, `left_hand`, `right_hand`, `torso`, `frame_rate`, `duration_sec`, `stimming_detected` (coordinate arrays plus scalar metadata). Audio is WAV/RIFF. Labels and tags are useful filters; description, label, tags, search_text, IDs, severity, status, and modality names support text retrieval. Numeric signal arrays are retained as source files and not flattened into misleading prose.

## Architecture

```text
Browser (React + Vite)
   └── FastAPI JSON API
       ├── data_loader → CSV + JSON metadata
       ├── document_builder → one sample document + 4 file references
       ├── preprocessing → Unicode normalization, lowercasing, tokenization, stopword removal
       ├── BooleanModel → term-to-document inverted index + AND/OR/NOT
       ├── VectorModel → TF-IDF document/query vectors + cosine ranking
       ├── ProbabilisticModel → smoothed query likelihood + heap-based probability ranking
       ├── BinaryIndependenceModel → binary term evidence + RSJ relevance weights
       ├── BM25Model → IDF + saturated term frequency + document-length normalization
       ├── OverlapListModel → nested sample/metadata regions + region-aware postings
       ├── NonOverlapListModel → disjoint metadata regions + field postings
       └── ProximalNodesModel → record/field/sentence hierarchy + positional postings
```

One IR document is one `sample_id` because labels, descriptions, tags, and the precomputed search text are sample-level. Its per-modality file objects retain indexed path, resolved on-disk path, filename, type, and existence. The implementation reads only the supplied data; it does not create an index file in or modify the dataset.

## Install and run

Backend (Python 3.10+):

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

The default dataset location is this repository's `Dataset for autism copy`. Override it with `AUTISM_DATASET_DIR=/path/to/dataset`. Set `CORS_ORIGINS` to comma-separated frontend origins if needed. Interactive API documentation is at `http://localhost:8000/docs`.

Frontend (Node.js):

```sh
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` before starting Vite to point to a non-default backend (default `http://localhost:8000`).

## API

- `GET /health` — process and loaded document count.
- `GET /dataset/stats` — totals, available modalities, labels, file types, missing references.
- `POST /search/boolean` — `{"query":"autism AND eye","modality":"all","top_k":10}`. Supports terms, parentheses, AND, OR, NOT (NOT > AND > OR). Missing terms yield no matches; invalid/empty syntax returns HTTP 400.
- `POST /search/vector` — `{"query":"child autism screening","modality":"image","top_k":10}`. Returns positive cosine matches ranked descending; unknown words contribute no weight.
- `POST /search/probabilistic` — `{"query":"child autism screening","modality":"all","top_k":10}`. Returns candidates ranked by `P(document | query)` with the probability exposed in both `probability` and `score`.
- `POST /search/bim` — accepts the same query, modality, and Top-K fields, plus optional `relevant_ids` and `non_relevant_ids` arrays for query-specific relevance feedback. Without feedback, it uses a collection-frequency initial estimate; its `score` is an RSJ weight, not a probability.
- `POST /search/bm25` — `{"query":"child autism screening","modality":"all","top_k":10}`. Returns documents ranked by BM25 score; scores are ranking values, not probabilities.
- `POST /search/overlap-lists` — `{"query":"child autism screening","modality":"all","top_k":10}`. Searches nested sample and metadata-field regions and returns matched terms, matched regions, and a region-coverage count. The count is an explanatory ordering value, not a probability or canonical relevance score.
- `POST /search/non-overlap-lists` — same request shape. Searches each metadata field as a separate disjoint region and returns matched terms, matched fields, and a field-coverage count. The count is an explanatory ordering value, not a probability or canonical relevance score.
- `POST /search/proximal-nodes` — same request shape. Searches record, field, and description-sentence nodes using token positions. Results favor nodes that contain more query terms in a tighter positional window and report the best node, its parent, and the window width in tokens.
- `GET /document/{document_id}` — complete sample record and file details.
- `GET /file/{document_id}/{modality}` — streams the source file when present, including image/audio previews.

Modality filters are `all`, `image`, `voice`, `motion`, and `physio`. Search terms are case-insensitive. Preprocessing lowercases, applies Unicode NFKC, tokenizes alphanumeric terms, removes common grammatical stopwords, and retains terms such as autism, ASD, gaze, speech, severity labels, numbers, and negation (`not`, `no`).

## Retrieval and evaluation

The Boolean model stores each normalized term's document-ID set. Operators combine sets; NOT complements against the 100-document universe. Vector retrieval uses log term frequency `(1 + ln(tf))`, smoothed IDF `ln((1+N)/(1+df))+1`, and cosine similarity. The query-likelihood model calculates `P(query | document)` with a multinomial language model and Laplace smoothing (`alpha=1`), then normalizes likelihoods with a uniform prior to estimate `P(document | query)`. The Binary Independence Model (BIM) uses binary term presence and Robertson-Sparck Jones weights. It starts with collection-frequency estimates and can update term probabilities from query-specific relevant and non-relevant sample IDs supplied to `/search/bim`. BM25 uses inverse document frequency, term-frequency saturation, and document-length normalization with `k1=1.5` and `b=0.75`. BIM and BM25 scores are ranking weights, not probabilities. Ranked models use a heap-backed top-K selection, except the existing vector model which sorts its matching candidates. Since the dataset provides no judged relevance ground truth, retrieval scores must not be presented as evaluation results. `backend/evaluation.py` computes precision, recall, F1, Precision@K, and Recall@K from an explicitly curated query-to-relevant-ID mapping; its BIM evaluation uses the initial estimate to avoid using evaluation judgments as training feedback. Useful query candidates based on actual metadata include `mild_asd`, `autism AND voice`, `typical OR moderate_asd`, and `severe ASD level 3`.

## UI and integration

The interface has two dedicated search pages: `/` for unstructured models (Boolean, vector space, query likelihood, BIM, and BM25) and `/#structured` for structured models (Overlap Lists, Non-Overlap Lists, and Proximal Nodes). Each page shows only its relevant model controls. Both share modality and Top-K filters, dataset counts, ranked result cards, match explanations, recent-search history, and a record modal with metadata and available file previews. The overlap-list model treats each sample as a parent region containing separate label, description, severity, status, tag, and modality regions; the non-overlap-list model indexes those fields separately without a parent region. The Proximal Nodes model adds sentence nodes under descriptions and uses token positions to favor query terms that occur near each other in a node. BIM result cards let a user mark samples relevant or not relevant and rerank with that feedback. Its blue, teal, violet, coral, and gold accents draw on the neurodiversity infinity spectrum while keeping high-contrast text and controls. Search results describe dataset records and are not clinical screening decisions.

Run the repository checks with `python3 -m unittest discover -s backend/tests -v`. The test suite exercises data loading/building, preprocessing, Boolean operators and invalid queries, vector scores/top-K, modality filters, empty results, and evaluation metrics. FastAPI endpoint schemas are documented in `/docs` and endpoint smoke checks require installing backend requirements first.
